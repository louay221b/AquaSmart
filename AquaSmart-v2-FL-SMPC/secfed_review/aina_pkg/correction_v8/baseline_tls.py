"""Minimal unmasked mTLS sum using the V7 transport/orchestrator.
For sequential benchmark calls only. Server intentionally sees clear updates.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'correction_v7'/'vendor'))
import numpy as np
import socket,time
from fl_smpc import tls_secagg_v45 as tls


def _plain_serve(raw,addr,context,state,timeout):
    who=None
    try:
        with context.wrap_socket(raw,server_side=True) as sock:
            sock.settimeout(timeout)
            who=tls._subject_cn(sock.getpeercert())
            obj,_=tls._recv_frame(sock)
            if who not in state.ids or obj.get('kind')!='plain_upload' or obj.get('name')!=who or obj.get('session')!=state.session:
                raise ValueError('Invalid identity/session')
            v=np.asarray(obj['vector'],dtype=float)
            if v.shape!=(state.dimension,) or not np.isfinite(v).all():raise ValueError('Invalid vector')
            with state.condition:
                if who in state.uploads:raise ValueError('Duplicate upload')
                state.uploads[who]=v
                if len(state.uploads)==len(state.ids):
                    state.outcome=(np.array([state.uploads[n] for n in state.ids]).sum(axis=0).tolist(),{'session':state.session,'mode':'unmasked'})
                state.condition.notify_all()
            state.wait(lambda:state.outcome is not None,timeout)
            with state.condition:vec,receipt=state.outcome
            tls._send_frame(sock,{'kind':'aggregate','sum':vec,'receipt':receipt})
    except Exception as e:state.fail(f'Plain worker {who}: {type(e).__name__}: {e}')
    finally:raw.close()


def plain_server(*args):
    # Patch ONLY inside spawned server. Reuse listener, mTLS, threading and reporting.
    tls._serve_client=_plain_serve
    tls._server_worker(*args)


def plain_client(pipe,name,vector,missing,ca,cert,key,port,ids,dim,session,timeout,fault):
    start=time.perf_counter();sent=received=0
    try:
        if missing or fault:raise ValueError('Baseline benchmark is all-online only')
        context=tls._client_tls_context(ca,cert,key)
        with socket.create_connection((tls.TLS_HOST,port),timeout=timeout) as raw:
            raw.settimeout(timeout)
            with context.wrap_socket(raw,server_hostname='localhost') as sock:
                sock.settimeout(timeout)
                sent=tls._send_frame(sock,{'kind':'plain_upload','name':name,'session':session,'vector':vector})
                msg,received=tls._recv_frame(sock)
                if msg.get('kind')!='aggregate' or msg['receipt']['session']!=session:raise ValueError('Invalid aggregate')
                pipe.send({'sum':msg['sum'],'receipt':msg['receipt'],'sent':sent,'received':received,'messages_out':1,'messages_in':1,'wall_sec':time.perf_counter()-start,'tls_version':sock.version()})
    except Exception as e:pipe.send({'error':f'{type(e).__name__}: {e}','sent':sent,'received':received})
    finally:pipe.close()


def plain_sum_round(updates,timeout_sec=30.):
    # Swap process targets only in benchmark parent; restore on exit.
    # Same V7 PKI, spawn, pipes, framing, deadlines and cleanup.
    old=(tls._server_worker,tls._client_worker)
    try:
        tls._server_worker,tls._client_worker=plain_server,plain_client
        return tls.tls_secure_sum_round(updates,timeout_sec=timeout_sec)
    finally:tls._server_worker,tls._client_worker=old
