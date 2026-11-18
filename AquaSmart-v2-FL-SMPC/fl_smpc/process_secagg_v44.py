"""V4.4 OS-process isolation demo for dropout-tolerant secure aggregation.

Separate spawned client workers each hold X25519+Ed25519 private keys. A
separate spawned aggregator worker receives only public keys, signatures and
masked/recovery messages. The experiment parent relays JSON bytes over local
multiprocessing Pipes. This is NOT TCP traffic, TLS, deployment, or a claim
that the experiment orchestrator (which initially holds vectors) is private.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import secrets
import time
from typing import Mapping, Sequence

import numpy as np

from fl_smpc.secure_aggregation import canonical, SecureAggregationError, DropoutError
from fl_smpc.secure_aggregation_v44 import RecoverableClient, RecoverableServer


def _json_send(conn, payload):
    raw=canonical(payload)
    conn.send_bytes(raw)
    return len(raw)


def _json_recv(conn, timeout):
    if not conn.poll(timeout):
        raise TimeoutError('Process worker did not respond within timeout')
    raw=conn.recv_bytes()
    data=json.loads(raw)
    if data.get('error'):
        raise SecureAggregationError(data['error'])
    return data,len(raw)


def _client_worker(conn, name, weighted_update):
    client=RecoverableClient(name)
    try:
        _json_send(conn,{'kind':'ready','client':name,
                         'key':client.identity_public_key.hex()})
        while True:
            try:
                request=json.loads(conn.recv_bytes())
            except EOFError:
                break
            cmd=request.get('cmd')
            if cmd=='stop': break
            try:
                if cmd=='announce':
                    value=client.announce(request['session'])
                elif cmd=='upload':
                    keys={k:bytes.fromhex(v) for k,v in request['keys'].items()}
                    value=client.upload(weighted_update,request['announcements'],keys,
                                        request['active'],request['dimension'])
                elif cmd=='recover':
                    keys={k:bytes.fromhex(v) for k,v in request['keys'].items()}
                    value=client.recover(request['announcements'],request['uploads'],keys,
                                         request['active'],request['dimension'])
                else:
                    raise SecureAggregationError('Unknown client action')
                _json_send(conn,{'kind':'ok','value':value})
            except Exception as exc:
                _json_send(conn,{'error':f'{type(exc).__name__}: {exc}'})
    finally:
        client.finish()
        conn.close()


def _server_worker(conn, keys_hex, active, dimension):
    server=RecoverableServer({k:bytes.fromhex(v) for k,v in keys_hex.items()},
                             active,dimension)
    try:
        _json_send(conn,{'kind':'ready'})
        while True:
            try: req=json.loads(conn.recv_bytes())
            except EOFError: break
            if req.get('cmd')=='stop': break
            try:
                if req.get('cmd')!='aggregate':
                    raise SecureAggregationError('Unsupported server action')
                agg,receipt=server.aggregate(req['announcements'],req['uploads'],
                                             req['recovery'],req['session'])
                _json_send(conn,{'kind':'ok','sum':agg.tolist(),'receipt':receipt})
            except Exception as exc:
                _json_send(conn,{'error':f'{type(exc).__name__}: {exc}'})
    finally:
        conn.close()


def process_secure_sum(updates: Mapping[str, Sequence[float]], *, missing=(),
                       timeout_sec: float=40.0, session=None):
    """Run a single secure SUM; simulate missing AFTER key announcement.

    Process-spawn (Windows compatible) measured bytes represent JSON payloads
    sent via Pipes, including setup/control but not process-init arguments,
    process bootstrap or OS IPC headers. Only survivors' updates are aggregated.
    """
    names=sorted(updates)
    if len(names)<2: raise ValueError('At least 2 clients needed')
    dropout=set(missing)
    if not dropout.issubset(names): raise ValueError('Unknown dropout name')
    if len(names)-len(dropout)<2: raise DropoutError('Need 2 or more survivors')
    dimension=len(updates[names[0]])
    for value in updates.values():
        a=np.asarray(value,dtype=np.float64)
        if a.shape!=(dimension,) or not np.isfinite(a).all():
            raise ValueError('Invalid weighted vector shape/values')
    session=session or secrets.token_hex(24)
    ctx=mp.get_context('spawn')
    workers=[]
    client_links={}
    byte_count={'client_to_coordinator':0,'coordinator_to_client':0,
                'server_to_coordinator':0,'coordinator_to_server':0}
    message_count=0
    t0=time.perf_counter()
    def send(conn,packet,channel):
        nonlocal message_count
        byte_count[channel] += _json_send(conn,packet)
        message_count += 1
    def recv(conn,channel):
        nonlocal message_count
        obj,n=_json_recv(conn,timeout_sec)
        byte_count[channel] += n
        message_count += 1
        return obj
    try:
        for name in names:
            parent,child=ctx.Pipe(duplex=True)
            proc=ctx.Process(target=_client_worker,args=(child,name,list(updates[name])))
            proc.start();child.close()
            workers.append((proc,parent))
            client_links[name]=parent
        keys_hex={}
        for name in names:
            r=recv(client_links[name],'client_to_coordinator')
            if r['kind']!='ready' or r['client']!=name: raise RuntimeError('Worker identity mismatch')
            keys_hex[name]=r['key']
        parent,child=ctx.Pipe(duplex=True)
        serv=ctx.Process(target=_server_worker,args=(child,keys_hex,names,dimension))
        serv.start();child.close()
        workers.append((serv,parent))
        if recv(parent,'server_to_coordinator')['kind']!='ready':
            raise RuntimeError('Aggregator startup failed')
        announcements=[]
        for name in names:
            send(client_links[name],{'cmd':'announce','session':session},
                 'coordinator_to_client')
            announcements.append(recv(client_links[name],'client_to_coordinator')['value'])
        for name in sorted(dropout):
            # Models a client disconnecting after signed public-key announcement.
            send(client_links[name],{'cmd':'stop'},'coordinator_to_client')
        uploads=[]
        for name in names:
            if name in dropout:continue
            send(client_links[name],{'cmd':'upload','announcements':announcements,
                 'keys':keys_hex,'active':names,'dimension':dimension},'coordinator_to_client')
            uploads.append(recv(client_links[name],'client_to_coordinator')['value'])
        proofs=[]
        for name in names:
            if name in dropout: continue
            send(client_links[name],{'cmd':'recover','announcements':announcements,
                 'uploads':uploads,'keys':keys_hex,'active':names,'dimension':dimension},
                 'coordinator_to_client')
            proof=recv(client_links[name],'client_to_coordinator')['value']
            if proof is not None: proofs.append(proof)
        send(parent,{'cmd':'aggregate','session':session,'announcements':announcements,
                     'uploads':uploads,'recovery':proofs},'coordinator_to_server')
        res=recv(parent,'server_to_coordinator')
        out=np.asarray(res['sum'],dtype=np.float64)
        receipt=res['receipt']
        return out, receipt, {
            'ipc_payload_bytes':sum(byte_count.values()),
            'ipc_payload_breakdown':byte_count,
            'ipc_messages':message_count,
            'round_wall_sec':time.perf_counter()-t0,
            'server_verify_sum_sec':receipt['verify_sum_sec'],
            'transport':'local spawned OS processes / JSON multiprocessing.Pipe',
            'communication_scope':'measured JSON Pipe payload; excludes vector process-start arguments, IPC framing, OS process startup traffic and TCP/TLS',
            'security_scope':'honest-but-curious non-colluding server; honest clients; >=2 survivors; pre-upload dropout only',
        }
    finally:
        for proc, conn in workers:
            try:
                if proc.is_alive(): _json_send(conn,{'cmd':'stop'})
            except (BrokenPipeError,OSError,EOFError): pass
        for proc,conn in workers:
            try:conn.close()
            except OSError:pass
            proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate();proc.join(timeout=1)