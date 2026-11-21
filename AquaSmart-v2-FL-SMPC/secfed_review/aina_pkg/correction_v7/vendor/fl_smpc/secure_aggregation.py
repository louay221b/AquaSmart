"""AquaSmart V4.3 — authenticated cryptographic pairwise secure SUM (prototype).

Threat model: honest clients, honest-but-curious aggregation server, ALL clients
online; non-colluding server. Identity verification keys must be provisioned
out of band. This is an in-process protocol demonstration, not distributed
network service or dropout-tolerant, malicious-secure, differential privacy,
or an audited implementation of Bonawitz et al. secure aggregation.

The server sees signed X25519 public announcements and signed, masked modular
vectors, never individual plaintext vectors at its protocol API boundary.
Pairwise shared secrets are derived by X25519 and HKDF, masks via ChaCha20.
Fixed-point quantization and modular arithmetic preserve an approximate sum.
If even one participant drops, the round FAILS CLOSED with no aggregate.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import struct
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VERSION = 'aquasmart-secagg-v43-prototype-1'
PRIME = (1 << 61) - 1
SCALE = 100_000_000
MAX_ABS = 10_000.0
MAX_DIMENSION = 2048


class SecureAggregationError(ValueError):
    """Protocol refused a malformed/unauthenticated/unsafe round."""


class DropoutError(SecureAggregationError):
    """A participant is unavailable: abort rather than reveal pairwise masks."""


def canonical(value: dict | list) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _hash(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _verify_signature(pub: bytes, domain: bytes, body: dict, signature_hex: str):
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(pub).verify(
            bytes.fromhex(signature_hex), domain + canonical(body))
    except (ValueError, InvalidSignature, TypeError) as exc:
        raise SecureAggregationError('Invalid authenticated client signature') from exc


def _validated_roster(announcements: Sequence[dict], trusted_keys: Mapping[str, bytes],
                      expected_ids: Sequence[str], session_id: str, dimension: int,
                      scale: int = SCALE):
    if not session_id or not isinstance(session_id, str):
        raise SecureAggregationError('Missing round session identifier')
    if dimension <= 0 or dimension > MAX_DIMENSION or dimension != int(dimension):
        raise SecureAggregationError('Invalid model dimension')
    if scale <= 0 or len(expected_ids) * MAX_ABS * scale >= PRIME // 2:
        raise SecureAggregationError('Unsafe quantization parameters: modular wrap risk')
    if len(set(expected_ids)) != len(expected_ids) or len(expected_ids) < 2:
        raise SecureAggregationError('Need >= 2 unique clients')
    if set(expected_ids) != set(trusted_keys):
        raise SecureAggregationError('Trust registry differs from expected roster')
    if len(announcements) != len(expected_ids):
        raise DropoutError('Missing client key announcement: secure round aborted')
    roster = {}
    for packet in announcements:
        body = {k: packet.get(k) for k in ('version', 'round', 'client', 'dh_pub')}
        name = body['client']
        if name not in trusted_keys or name in roster:
            raise SecureAggregationError('Unknown or duplicate announcement')
        if body['version'] != VERSION or body['round'] != session_id:
            raise SecureAggregationError('Wrong protocol version or session')
        try:
            pub = bytes.fromhex(body['dh_pub'])
            x25519.X25519PublicKey.from_public_bytes(pub)
        except (TypeError, ValueError) as exc:
            raise SecureAggregationError('Invalid X25519 key announcement') from exc
        _verify_signature(trusted_keys[name], b'AquaSmart:dh:v43:', body, packet.get('signature', ''))
        roster[name] = dict(body, signature=packet['signature'])
    if set(roster) != set(expected_ids):
        raise DropoutError('Missing authorized announcement')
    signed_roster = [roster[name] for name in sorted(roster)]
    transcript = {'version': VERSION, 'round': session_id, 'dimension': dimension,
                  'scale': scale, 'prime': PRIME, 'announcements': signed_roster}
    return signed_roster, _hash(transcript)


def _mask_values(secret: bytes, transcript_digest: str, left: str, right: str,
                 dimension: int) -> list[int]:
    if not left < right:
        raise SecureAggregationError('Peer IDs must be ordered')
    salt = bytes.fromhex(transcript_digest)
    # Per-pair + per-round distinct ChaCha20 keys; 16-byte zero nonce is safe
    # because each key is used once (fresh ephemeral X25519 keys each round).
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
               info=b'AquaSmart:pair-mask:v43:' + canonical([left, right])).derive(secret)
    stream = Cipher(algorithms.ChaCha20(key, b'\0' * 16), mode=None).encryptor()
    raw = stream.update(b'\0' * (dimension * 8)) + stream.finalize()
    return [v % PRIME for (v,) in struct.iter_unpack('<Q', raw)]


def quantize(vector: np.ndarray | Sequence[float], dimension: int,
             scale: int = SCALE) -> list[int]:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1 or arr.size != dimension or not np.all(np.isfinite(arr)):
        raise SecureAggregationError('Non-finite or wrong-dimensional client update')
    if np.any(np.abs(arr) > MAX_ABS):
        raise SecureAggregationError('Update exceeds agreed fixed-point magnitude bound')
    return [int(round(float(x) * scale)) % PRIME for x in arr]


class SecureClient:
    """Client holds Ed25519 signing secret and per-round X25519 private key."""

    def __init__(self, client_id: str):
        if not client_id or not isinstance(client_id, str):
            raise ValueError('client_id must be nonempty')
        self.client_id = client_id
        self.__signing = ed25519.Ed25519PrivateKey.generate()
        self.__dh = None
        self.__round = None
        self.__used = False
        self.last_mask_crypto_sec = 0.0

    @property
    def identity_public_key(self) -> bytes:
        return self.__signing.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)

    def start_round(self, session_id: str) -> dict:
        if not session_id or not isinstance(session_id, str):
            raise SecureAggregationError('Missing round ID')
        self.__dh = x25519.X25519PrivateKey.generate()
        self.__round = session_id
        self.__used = False
        self.last_mask_crypto_sec = 0.0
        body = {'version': VERSION, 'round': session_id, 'client': self.client_id,
                'dh_pub': self.__dh.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw).hex()}
        signature = self.__signing.sign(b'AquaSmart:dh:v43:' + canonical(body)).hex()
        return dict(body, signature=signature)

    def mask_update(self, weighted_update: np.ndarray, announcements: Sequence[dict],
                    trusted_keys: Mapping[str, bytes], expected_ids: Sequence[str],
                    dimension: int, scale: int = SCALE) -> dict:
        if self.__dh is None or self.__round is None or self.__used:
            raise SecureAggregationError('No fresh key for this round or duplicate upload')
        crypto_start = time.perf_counter()
        if self.client_id not in expected_ids:
            raise SecureAggregationError('Client is not in authorized roster')
        roster, transcript = _validated_roster(
            announcements, trusted_keys, expected_ids, self.__round, dimension, scale)
        mine = next(packet for packet in roster if packet['client'] == self.client_id)
        my_public = self.__dh.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw).hex()
        if mine['dh_pub'] != my_public:
            raise SecureAggregationError('Server substituted the client\'s own key')
        residues = quantize(weighted_update, dimension, scale)
        for peer in roster:
            peer_id = peer['client']
            if peer_id == self.client_id:
                continue
            dh_pub = x25519.X25519PublicKey.from_public_bytes(bytes.fromhex(peer['dh_pub']))
            shared = self.__dh.exchange(dh_pub)
            left, right = sorted([self.client_id, peer_id])
            mask = _mask_values(shared, transcript, left, right, dimension)
            if self.client_id == left:
                residues = [(x + m) % PRIME for x, m in zip(residues, mask)]
            else:
                residues = [(x - m) % PRIME for x, m in zip(residues, mask)]
        body = {'version': VERSION, 'round': self.__round, 'client': self.client_id,
                'transcript': transcript, 'masked': residues}
        signature = self.__signing.sign(b'AquaSmart:update:v43:' + canonical(body)).hex()
        self.__used = True
        self.__dh = None
        self.last_mask_crypto_sec = time.perf_counter() - crypto_start
        return dict(body, signature=signature)

    def train_and_mask(self, X, y, current_model, local_epochs, lr, alpha, rng,
                       client_weight, announcements, trusted_keys, expected_ids,
                       dimension: int):
        """Local computation returns only signed masked update to coordinator.

        For benchmark equivalence uses same SGD rng schedule as V4.2. In this
        single-process harness datasets/rng still share an address space.
        """
        from fl_smpc.common import gd_update, get_params
        w, b = current_model
        new_w, new_b = gd_update(X, y, w.copy(), float(b), epochs=local_epochs,
                                 lr=lr, alpha=alpha, rng=rng)
        weighted = float(client_weight) * get_params(new_w, new_b)
        return self.mask_update(weighted, announcements, trusted_keys,
                                expected_ids, dimension)


@dataclass(frozen=True)
class WireStats:
    key_upload_bytes: int
    roster_broadcast_bytes: int
    masked_upload_bytes: int
    global_model_broadcast_bytes_estimate: int
    message_count: int
    key_agreement_and_mask_sec: float
    server_verify_and_sum_sec: float

    @property
    def total_bytes(self):
        return (self.key_upload_bytes + self.roster_broadcast_bytes
                + self.masked_upload_bytes + self.global_model_broadcast_bytes_estimate)

    def as_dict(self):
        return dict(self.__dict__, total_bytes=self.total_bytes,
                    note='Serialized in-process messages, NOT actual network traffic or latency; model broadcast estimated')


class SecureServer:
    """Aggregation server stores only authenticated public verification keys."""

    def __init__(self, trusted_keys: Mapping[str, bytes], dimension: int,
                 scale: int = SCALE):
        self.trusted_keys = dict(trusted_keys)
        self.expected_ids = tuple(sorted(self.trusted_keys))
        self.dimension = dimension
        self.scale = scale
        self._completed = set()
        if len(self.expected_ids) < 2 or dimension < 1 or dimension > MAX_DIMENSION:
            raise SecureAggregationError('Invalid server configuration')
        if len(self.expected_ids) * MAX_ABS * scale >= PRIME // 2:
            raise SecureAggregationError('Potential modular overflow')

    def aggregate(self, announcements: Sequence[dict], updates: Sequence[dict],
                  session_id: str) -> tuple[np.ndarray, dict]:
        begin = time.perf_counter()
        if session_id in self._completed:
            raise SecureAggregationError('Replayed completed session')
        roster, transcript = _validated_roster(
            announcements, self.trusted_keys, self.expected_ids,
            session_id, self.dimension, self.scale)
        if len(updates) != len(self.expected_ids):
            raise DropoutError('Missing masked update: round aborted, no aggregate output')
        collected = {}
        for msg in updates:
            body = {k: msg.get(k) for k in ('version', 'round', 'client', 'transcript', 'masked')}
            client_id = body['client']
            if client_id not in self.trusted_keys or client_id in collected:
                raise SecureAggregationError('Unknown or duplicate client update')
            if body['version'] != VERSION or body['round'] != session_id or body['transcript'] != transcript:
                raise SecureAggregationError('Update belongs to another session or roster')
            residues = body['masked']
            if not isinstance(residues, list) or len(residues) != self.dimension or any(
                    type(z) is not int or z < 0 or z >= PRIME for z in residues):
                raise SecureAggregationError('Invalid masked modular vector')
            _verify_signature(self.trusted_keys[client_id], b'AquaSmart:update:v43:',
                              body, msg.get('signature', ''))
            collected[client_id] = residues
        if set(collected) != set(self.expected_ids):
            raise DropoutError('Incomplete authenticated roster; aborted')
        acc = [0] * self.dimension
        for c in sorted(collected):
            for j, val in enumerate(collected[c]):
                acc[j] = (acc[j] + val) % PRIME
        signed = [z if z <= PRIME // 2 else z - PRIME for z in acc]
        result = np.array([float(z) / self.scale for z in signed], dtype=np.float64)
        if np.any(np.abs(result) > len(self.expected_ids) * MAX_ABS + 1):
            raise SecureAggregationError('Aggregate outside numerical safety bounds')
        self._completed.add(session_id)
        server_sec = time.perf_counter() - begin
        return result, {'session_id_hash': hashlib.sha256(session_id.encode()).hexdigest()[:12],
                        'transcript_hash': transcript,
                        'n_clients': len(self.expected_ids),
                        'server_verify_and_sum_sec': server_sec}


def secure_sum_round(clients: Mapping[str, SecureClient],
                     weighted_updates: Mapping[str, Sequence[float]],
                     *, server: SecureServer | None = None,
                     session_id: str | None = None):
    """In-memory harness for an authenticated all-online secure SUM round.

    The harness sees plaintext inputs for testing. A real deployment must
    isolate SecureClient on separate hosts/processes with secure transport.
    """
    if server is None:
        server = SecureServer({k: c.identity_public_key for k, c in clients.items()},
                              len(next(iter(weighted_updates.values()))))
    ids = server.expected_ids
    if set(clients) != set(ids) or set(weighted_updates) != set(ids):
        raise DropoutError('Roster incomplete: no partial aggregation')
    session_id = session_id or secrets.token_hex(24)
    begin = time.perf_counter()
    announcements = [clients[k].start_round(session_id) for k in ids]
    uploads = [clients[k].mask_update(weighted_updates[k], announcements,
                                     server.trusted_keys, ids, server.dimension,
                                     server.scale) for k in ids]
    client_sec = time.perf_counter() - begin
    aggregate, receipt = server.aggregate(announcements, uploads, session_id)
    stats = _wire_stats(announcements, uploads, server.dimension, client_sec,
                        receipt['server_verify_and_sum_sec'])
    return aggregate, receipt, stats, (announcements, uploads)


def _wire_stats(announcements, uploads, dimension, client_sec, server_sec):
    n = len(announcements)
    roster = sorted(announcements, key=lambda item: item['client'])
    return WireStats(
        key_upload_bytes=sum(len(canonical(p)) for p in announcements),
        roster_broadcast_bytes=n * len(canonical(roster)),
        masked_upload_bytes=sum(len(canonical(p)) for p in uploads),
        global_model_broadcast_bytes_estimate=n * dimension * 8,
        message_count=3*n+ n,  # keys up, roster down, masked vectors up, model down
        key_agreement_and_mask_sec=client_sec,
        server_verify_and_sum_sec=server_sec,
    )
