"""AquaSmart V4.4: pairwise secure SUM with recovery of *non-uploading* clients.

Security scope: honest clients, honest-but-curious coordinator, no collusion,
>=2 uploaded clients, authenticated identity keys pre-provisioned out of band.
A client can announce its ephemeral X25519 key then disappear BEFORE uploading.
Survivors disclose ONLY adjustments for pairwise masks with these missing peers.
Uncancelled masks BETWEEN surviving clients remain secret. A client that uploads
then disappears before producing a recovery proof causes the round to ABORT.

This is an experimental protocol, not an audited threshold/dropout protocol.
The coordinator MUST NOT falsely omit received uploads; this is an honest-but-
curious assumption (malicious-server privacy is not claimed). Do not deploy.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

from fl_smpc.secure_aggregation import (
    PRIME, SCALE, MAX_ABS, MAX_DIMENSION, SecureAggregationError, DropoutError,
    _verify_signature, _mask_values, canonical, quantize,
)

VERSION44 = 'aquasmart-secagg-v44-dropout-prototype-1'
_DH = b'AquaSmart:v44:announcement:'
_UPLOAD = b'AquaSmart:v44:upload:'
_RECOVERY = b'AquaSmart:v44:recovery:'


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def verify_roster(announcements: Sequence[dict], keys: Mapping[str, bytes],
                  active: Sequence[str], session: str, dimension: int,
                  scale: int = SCALE):
    active = tuple(sorted(active))
    if not session or not isinstance(session, str):
        raise SecureAggregationError('Missing session')
    if (len(active) < 2 or len(set(active)) != len(active)
            or not set(active).issubset(keys)):
        raise SecureAggregationError('Invalid authorized active roster')
    if dimension < 1 or dimension > MAX_DIMENSION or scale <= 0:
        raise SecureAggregationError('Invalid dimension or scale')
    if len(active) * MAX_ABS * scale >= PRIME // 2:
        raise SecureAggregationError('Possible fixed-point modular overflow')
    if len(announcements) != len(active):
        raise DropoutError('Missing authenticated announcement; start a NEW round')
    by_id = {}
    for item in announcements:
        body = {k: item.get(k) for k in ('version', 'round', 'client', 'dh_pub')}
        k = body['client']
        if k not in active or k in by_id:
            raise SecureAggregationError('Unknown/duplicate key announcement')
        if body['version'] != VERSION44 or body['round'] != session:
            raise SecureAggregationError('Stale/mismatched key announcement')
        try:
            pub = bytes.fromhex(body['dh_pub'])
            x25519.X25519PublicKey.from_public_bytes(pub)
        except (TypeError, ValueError) as exc:
            raise SecureAggregationError('Bad X25519 public key') from exc
        _verify_signature(keys[k], _DH, body, item.get('signature', ''))
        by_id[k] = item
    return [by_id[k] for k in active], digest({
        'version': VERSION44, 'round': session, 'dimension': dimension,
        'scale': scale, 'prime': PRIME, 'announcements': [by_id[k] for k in active],
    })


def verify_uploads(uploads, keys, active, transcript, session, dimension):
    uploaded = {}
    for msg in uploads:
        body = {k: msg.get(k) for k in ('version','round','client','transcript','masked')}
        who = body['client']
        if who not in active or who in uploaded:
            raise SecureAggregationError('Duplicate/unknown masked upload')
        if (body['version'] != VERSION44 or body['round'] != session
                or body['transcript'] != transcript):
            raise SecureAggregationError('Wrong version/session/transcript')
        q = body['masked']
        if (not isinstance(q, list) or len(q) != dimension or
                any(type(z) is not int or z < 0 or z >= PRIME for z in q)):
            raise SecureAggregationError('Malformed masked upload')
        _verify_signature(keys[who], _UPLOAD, body, msg.get('signature', ''))
        uploaded[who] = msg
    if len(uploaded) < 2:
        raise DropoutError('At least two client uploads required')
    return uploaded, digest([uploaded[k] for k in sorted(uploaded)])


class RecoverableClient:
    """All signing/DH private keys remain inside this client object/process."""

    def __init__(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError('Nonempty client identifier required')
        self.name = name
        self._sign = ed25519.Ed25519PrivateKey.generate()
        self._dh = None
        self._session = None
        self._state = 'idle'
        self._my_upload = None
        self._transcript = None

    @property
    def identity_public_key(self):
        return self._sign.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def announce(self, session):
        if self._state not in ('idle','done'):
            raise SecureAggregationError('Previous session unfinished')
        self._dh = x25519.X25519PrivateKey.generate()
        self._session = session
        self._state = 'announced'
        body = {'version': VERSION44, 'round':session,'client':self.name,
                'dh_pub':self._dh.public_key().public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()}
        return dict(body, signature=self._sign.sign(_DH + canonical(body)).hex())

    def upload(self, weighted_update, announcements, keys, active, dimension,
               scale=SCALE):
        if self._state != 'announced':
            raise SecureAggregationError('Client not announced or already uploaded')
        roster, transcript = verify_roster(
            announcements, keys, active, self._session, dimension, scale)
        mine = next(v for v in roster if v['client']==self.name)
        if mine['dh_pub'] != self._dh.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex():
            raise SecureAggregationError('Own ephemeral DH key substituted')
        values = quantize(weighted_update, dimension, scale)
        for peer in roster:
            other = peer['client']
            if other == self.name: continue
            shared = self._dh.exchange(x25519.X25519PublicKey.from_public_bytes(
                bytes.fromhex(peer['dh_pub'])))
            left,right = sorted((self.name,other))
            masks = _mask_values(shared, transcript, left, right, dimension)
            sign = 1 if self.name == left else -1
            values = [(a + sign*m) % PRIME for a,m in zip(values, masks)]
        body = {'version': VERSION44, 'round':self._session,'client':self.name,
                'transcript':transcript, 'masked':values}
        packet = dict(body, signature=self._sign.sign(_UPLOAD + canonical(body)).hex())
        self._state = 'uploaded'
        self._transcript = transcript
        self._my_upload = packet
        return packet

    def recover(self, announcements, uploads, keys, active, dimension, scale=SCALE):
        """Signed correction only for peers that NEVER uploaded their update.

        If no one dropped, nothing is disclosed. Server must genuinely follow
        the upload bookkeeping and present the same uploaded set to everyone.
        """
        if self._state != 'uploaded':
            raise DropoutError('Survivor must have uploaded before recovery')
        roster, transcript = verify_roster(
            announcements, keys, active, self._session, dimension, scale)
        uploaded, uploads_hash = verify_uploads(
            uploads, keys, active, transcript, self._session, dimension)
        if (self.name not in uploaded or uploaded[self.name] != self._my_upload
                or transcript != self._transcript):
            raise SecureAggregationError('Survivor upload missing or changed')
        missing = sorted(set(active) - set(uploaded))
        if not missing:
            self.finish()
            return None
        correction = [0] * dimension
        for item in roster:
            other = item['client']
            if other not in missing: continue
            shared = self._dh.exchange(x25519.X25519PublicKey.from_public_bytes(
                bytes.fromhex(item['dh_pub'])))
            left,right = sorted((self.name,other))
            masks = _mask_values(shared, transcript, left, right, dimension)
            # Remove surviving client's outstanding mask with absent peer.
            correction_sign = -1 if self.name == left else 1
            correction = [(a+correction_sign*m)%PRIME for a,m in zip(correction,masks)]
        body = {'version':VERSION44,'round':self._session,'client':self.name,
                'transcript':transcript,'uploads_hash':uploads_hash,
                'uploaded':sorted(uploaded),'missing':missing,'correction':correction}
        packet = dict(body, signature=self._sign.sign(_RECOVERY+canonical(body)).hex())
        self.finish()
        return packet

    def finish(self):
        self._dh = None
        self._state = 'done'
        self._my_upload = None
        self._transcript = None


class RecoverableServer:
    def __init__(self, trusted_keys, active, dimension, scale=SCALE):
        self.keys = dict(trusted_keys)
        self.active = tuple(sorted(active))
        self.dimension=dimension
        self.scale=scale
        self._used=set()
        if (len(self.active)<2 or not set(self.active).issubset(self.keys)
                or dimension<1 or dimension>MAX_DIMENSION):
            raise SecureAggregationError('Invalid server configuration')
        if len(self.active)*MAX_ABS*scale>=PRIME//2:
            raise SecureAggregationError('Possible modular overflow')

    def aggregate(self, announcements, uploads, recovery, session):
        start=time.perf_counter()
        if session in self._used: raise SecureAggregationError('Repeated completed session')
        roster,transcript=verify_roster(
            announcements,self.keys,self.active,session,self.dimension,self.scale)
        uploaded,uh=verify_uploads(
            uploads,self.keys,self.active,transcript,session,self.dimension)
        present=sorted(uploaded)
        missing=sorted(set(self.active)-set(uploaded))
        if missing and len(recovery)!=len(present):
            raise DropoutError('Missing recovery proof: secure round aborted')
        if not missing and recovery:
            raise SecureAggregationError('Unexpected recovery with no dropout')
        seen=set()
        corrections=[]
        for packet in recovery:
            body={k:packet.get(k) for k in ('version','round','client','transcript',
                    'uploads_hash','uploaded','missing','correction')}
            who=body['client']
            if who not in present or who in seen:
                raise SecureAggregationError('Recovery sender unknown/duplicate')
            if (body['version']!=VERSION44 or body['round']!=session or
                    body['transcript']!=transcript or body['uploads_hash']!=uh or
                    body['uploaded']!=present or body['missing']!=missing):
                raise SecureAggregationError('Incorrect recovery roster/digest')
            values=body['correction']
            if (not isinstance(values,list) or len(values)!=self.dimension or
                    any(type(z) is not int or z<0 or z>=PRIME for z in values)):
                raise SecureAggregationError('Malformed recovery values')
            _verify_signature(self.keys[who], _RECOVERY, body, packet.get('signature',''))
            seen.add(who)
            corrections.append(values)
        if missing and seen!=set(present):
            raise DropoutError('Not all survivors disclosed dropout-only corrections')
        acc=[0]*self.dimension
        for u in uploaded.values():
            acc=[(a+b)%PRIME for a,b in zip(acc,u['masked'])]
        for vec in corrections:
            acc=[(a+b)%PRIME for a,b in zip(acc,vec)]
        signed=[v if v<=PRIME//2 else v-PRIME for v in acc]
        result=np.array([float(v)/self.scale for v in signed],dtype=np.float64)
        if np.any(abs(result)>(len(present)*MAX_ABS+1)):
            raise SecureAggregationError('Unsafe recovered sum magnitude')
        self._used.add(session)
        return result, {'session_hash':hashlib.sha256(session.encode()).hexdigest()[:12],
                        'n_active':len(self.active),'n_uploaded':len(present),
                        'n_dropped':len(missing),'uploaded':present,'dropped':missing,
                        'transcript':transcript,'recovery_used':bool(missing),
                        'verify_sum_sec':time.perf_counter()-start}


def secure_sum_with_dropout(updates, *, missing=(), session=None):
    """Single-process numerical reference harness (NOT process isolated)."""
    names=sorted(updates)
    if not names: raise SecureAggregationError('No participants')
    dim=len(updates[names[0]])
    clients={k:RecoverableClient(k) for k in names}
    keys={k:v.identity_public_key for k,v in clients.items()}
    server=RecoverableServer(keys,names,dim)
    session=session or secrets.token_hex(16)
    if not set(missing).issubset(names):
        raise SecureAggregationError('Unknown dropout')
    announcements=[clients[k].announce(session) for k in names]
    uploads=[clients[k].upload(updates[k],announcements,keys,names,dim)
             for k in names if k not in missing]
    proofs=[clients[k].recover(announcements,uploads,keys,names,dim)
            for k in names if k not in missing]
    proofs=[p for p in proofs if p is not None]
    result,receipt=server.aggregate(announcements,uploads,proofs,session)
    return result,receipt,{'announcements':announcements,'uploads':uploads,'recovery':proofs}
