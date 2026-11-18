"""AquaSmart V4.5: loopback TCP/TLS 1.3 reference transport for V4.4 SecAgg.

Research / benchmark only. Processes are separated and authenticate using a
temporary test CA and client certificates. Each *client process* holds its
signing / DH private keys and masked update; the server receives signed
masked messages and dropout-recovery packets, never clear updates.

LIMITATIONS: The benchmark parent supplies each client's plaintext update to
its child process; the benchmark is not end-to-end private from that parent.
The coordinator is assumed honest-but-curious: it must honestly report all
received uploads, must not selectively omit them. No malicious-server,
collusion, late-dropout recovery, threshold cryptography, DP, or audit proof.
The measurement is TLS application frame bytes, not packet-capture wire bytes
(including TLS handshake overhead). This is NOT a production deployment.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import json
import multiprocessing as mp
import os
import secrets
import re
import socket
import ssl
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from fl_smpc.secure_aggregation import SecureAggregationError, DropoutError, _verify_signature, canonical
from fl_smpc.secure_aggregation_v44 import (
    RecoverableClient, RecoverableServer, _DH, verify_roster,
)

MAX_FRAME_BYTES = 2_000_000
TLS_HOST = "127.0.0.1"


def _send_frame(sock: ssl.SSLSocket, data: dict) -> int:
    encoded = canonical(data)
    if len(encoded) > MAX_FRAME_BYTES:
        raise SecureAggregationError("Oversized outbound TLS frame")
    packet = struct.pack("!I", len(encoded)) + encoded
    sock.sendall(packet)
    return len(packet)


def _receive_exact(sock: ssl.SSLSocket, n: int) -> bytes:
    pieces = []
    while n:
        block = sock.recv(n)
        if not block:
            raise EOFError("TLS connection closed before message completed")
        pieces.append(block)
        n -= len(block)
    return b"".join(pieces)


def _recv_frame(sock: ssl.SSLSocket):
    length = struct.unpack("!I", _receive_exact(sock, 4))[0]
    if length > MAX_FRAME_BYTES:
        raise SecureAggregationError("Oversized inbound TLS frame")
    raw = _receive_exact(sock, length)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SecureAggregationError("Expected JSON object")
    return value, 4 + length


def _subject_cn(cert: dict) -> str:
    for entry in cert.get("subject", ()):  # python ssl parsed peer cert
        for key, value in entry:
            if key == "commonName":
                return value
    raise SecureAggregationError("Peer TLS certificate lacks CN")


def _write_pem(path: Path, cert: x509.Certificate, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    cert_path = path.with_suffix(".cert.pem")
    key_path = path.with_suffix(".key.pem")
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return str(cert_path), str(key_path)


def _dev_pki(root: Path, names: list[str]):
    """Ephemeral CA in a temporary directory: test-only trust provisioning."""
    now = dt.datetime.now(dt.timezone.utc)
    before, after = now - dt.timedelta(minutes=5), now + dt.timedelta(days=1)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AquaSmart-v45-TEST-CA")])
    ca_cert = (
        x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(before).not_valid_after(after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                                     key_encipherment=False, data_encipherment=False,
                                     key_agreement=False, key_cert_sign=True, crl_sign=True,
                                     encipher_only=False, decipher_only=False), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = root / "test-ca.pem"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    credentials = {}
    for name in ["localhost", *names]:
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        usage = (ExtendedKeyUsageOID.SERVER_AUTH if name == "localhost"
                 else ExtendedKeyUsageOID.CLIENT_AUTH)
        b = (x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
             .public_key(key.public_key()).serial_number(x509.random_serial_number())
             .not_valid_before(before).not_valid_after(after)
             .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
             .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
             .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
             .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
             .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                                          key_encipherment=False, data_encipherment=False,
                                          key_agreement=False, key_cert_sign=False, crl_sign=False,
                                          encipher_only=False, decipher_only=False), critical=True))
        if name == "localhost":
            b = b.add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(TLS_HOST))
            ]), critical=False)
        credentials[name] = _write_pem(root / name, b.sign(ca_key, hashes.SHA256()), key)
    return str(ca_path), credentials


def _server_tls_context(ca: str, cert: str, key: str):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(cert, key)
    context.load_verify_locations(cafile=ca)
    context.verify_mode = ssl.CERT_REQUIRED  # mTLS
    return context


def _client_tls_context(ca: str, cert: str, key: str):
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(cert, key)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


@dataclass
class _State:
    ids: tuple[str, ...]
    dimension: int
    session: str
    condition: threading.Condition = field(default_factory=threading.Condition)
    announcements: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict)
    uploads: dict = field(default_factory=dict)
    dropped: set = field(default_factory=set)
    recovery: dict = field(default_factory=dict)
    failure: str | None = None
    outcome: tuple | None = None
    wall_begin: float = field(default_factory=time.perf_counter)

    def fail(self, error: str):
        with self.condition:
            if not self.failure:
                self.failure = error
            self.condition.notify_all()

    def wait(self, predicate, timeout: float):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: predicate() or self.failure is not None, timeout=timeout)
            if not ready:
                self.fail("Protocol phase timeout")
            if self.failure:
                raise DropoutError(self.failure)
            if not predicate():
                raise DropoutError("Protocol phase incomplete")


def _serve_client(raw: socket.socket, addr, context, state: _State, timeout: float):
    who = None
    uploaded = False
    try:
        with context.wrap_socket(raw, server_side=True) as sock:
            sock.settimeout(timeout)
            who = _subject_cn(sock.getpeercert())
            if who not in state.ids:
                raise SecureAggregationError("Unauthorized mTLS client")
            hello, _ = _recv_frame(sock)
            if hello.get("kind") != "announce" or hello.get("name") != who:
                raise SecureAggregationError("Wrong authenticated announce identity")
            announce, pub_hex = hello.get("packet"), hello.get("identity_pub")
            if not isinstance(announce, dict) or announce.get("client") != who:
                raise SecureAggregationError("Bad announcement")
            pub = bytes.fromhex(pub_hex)
            if len(pub) != 32:
                raise SecureAggregationError("Invalid Ed25519 verification key")
            body = {k: announce.get(k) for k in ("version", "round", "client", "dh_pub")}
            _verify_signature(pub, _DH, body, announce.get("signature", ""))
            with state.condition:
                if who in state.announcements:
                    raise SecureAggregationError("Duplicate TLS client connection")
                state.announcements[who] = announce
                state.keys[who] = pub
                state.condition.notify_all()
            state.wait(lambda: len(state.announcements) == len(state.ids), timeout)
            with state.condition:
                roster = [state.announcements[k] for k in state.ids]
                keys = {k: state.keys[k].hex() for k in state.ids}
            _send_frame(sock, {"kind": "roster", "announcements": roster, "keys": keys})
            try:
                msg, _ = _recv_frame(sock)
            except EOFError:
                with state.condition:
                    state.dropped.add(who)
                    state.condition.notify_all()
                return
            if msg.get("kind") != "upload":
                raise SecureAggregationError("Expected masked upload")
            upload = msg.get("packet")
            if not isinstance(upload, dict) or upload.get("client") != who:
                raise SecureAggregationError("Malformed upload identity")
            with state.condition:
                state.uploads[who] = upload
                uploaded = True
                state.condition.notify_all()
            state.wait(lambda: len(state.uploads) + len(state.dropped) == len(state.ids), timeout)
            with state.condition:
                if len(state.uploads) < 2:
                    raise DropoutError("At least two survivors required")
                uploads = [state.uploads[k] for k in sorted(state.uploads)]
            _send_frame(sock, {"kind": "uploads", "packets": uploads})
            message, _ = _recv_frame(sock)
            if message.get("kind") != "recovery":
                raise SecureAggregationError("Expected signed recovery or no-recovery ack")
            proof = message.get("packet")
            if proof is not None and (not isinstance(proof, dict) or proof.get("client") != who):
                raise SecureAggregationError("Invalid recovery sender")
            with state.condition:
                state.recovery[who] = proof
                if len(state.recovery) == len(state.uploads):
                    try:
                        server = RecoverableServer(state.keys, state.ids, state.dimension)
                        answer, receipt = server.aggregate(
                            [state.announcements[k] for k in state.ids],
                            [state.uploads[k] for k in sorted(state.uploads)],
                            [p for p in state.recovery.values() if p is not None],
                            state.session,
                        )
                        state.outcome = (answer.tolist(), receipt)
                    except Exception as e:
                        state.failure = f"Aggregation rejected: {type(e).__name__}: {e}"
                state.condition.notify_all()
            state.wait(lambda: state.outcome is not None, timeout)
            with state.condition:
                vector, receipt = state.outcome
            _send_frame(sock, {"kind": "aggregate", "sum": vector, "receipt": receipt})
    except Exception as e:
        state.fail(f"Client {who or addr} aborted: {type(e).__name__}: {e}")
    finally:
        try:
            raw.close()
        except OSError:
            pass


def _server_worker(pipe, ca, cert, key, ids, dim, session, timeout):
    listener = None
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((TLS_HOST, 0))
        listener.listen(len(ids) + 2)
        listener.settimeout(timeout)
        context = _server_tls_context(ca, cert, key)
        state = _State(tuple(sorted(ids)), dim, session)
        pipe.send({"port": listener.getsockname()[1]})
        threads = []
        for _ in ids:
            raw, addr = listener.accept()
            t = threading.Thread(target=_serve_client, args=(raw, addr, context, state, timeout), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=timeout * 5)
            if t.is_alive():
                state.fail("Worker thread still active after deadline")
        if state.failure:
            pipe.send({"error": state.failure})
        elif state.outcome:
            pipe.send({"aggregate": state.outcome[0], "receipt": state.outcome[1]})
        else:
            pipe.send({"error": "Missing aggregate"})
    except Exception as e:
        pipe.send({"error": f"Server failure: {type(e).__name__}: {e}"})
    finally:
        if listener:
            listener.close()
        pipe.close()


def _client_worker(pipe, name, vector, missing, ca, cert, key, port, ids, dim, session, timeout, fault):
    sent = received = messages_out = messages_in = 0
    start = time.perf_counter()
    try:
        client = RecoverableClient(name)
        context = _client_tls_context(ca, cert, key)
        with socket.create_connection((TLS_HOST, port), timeout=timeout) as plain:
            plain.settimeout(timeout)
            with context.wrap_socket(plain, server_hostname="localhost") as sock:
                sock.settimeout(timeout)
                def send(obj):
                    nonlocal sent, messages_out
                    sent += _send_frame(sock, obj)
                    messages_out += 1
                def recv():
                    nonlocal received, messages_in
                    obj, n = _recv_frame(sock)
                    received += n
                    messages_in += 1
                    return obj
                announce = client.announce(session)
                send({"kind": "announce", "name": name,
                      "identity_pub": client.identity_public_key.hex(), "packet": announce})
                roster = recv()
                if roster.get("kind") != "roster":
                    raise SecureAggregationError("Expected roster broadcast")
                keys = {k: bytes.fromhex(v) for k, v in roster["keys"].items()}
                announcements = roster["announcements"]
                if set(keys) != set(ids):
                    raise SecureAggregationError("Roster identity set changed")
                if missing:
                    # Truly close the TLS connection without sending an upload.
                    pipe.send({"dropped": name, "sent": sent, "received": received,
                               "messages_out": messages_out, "messages_in": messages_in,
                               "wall_sec": time.perf_counter() - start})
                    return
                update = client.upload(np.asarray(vector, dtype=np.float64),
                                       announcements, keys, ids, dim)
                if fault == "tamper_upload":
                    update["signature"] = "00" * 64
                send({"kind": "upload", "packet": update})
                if fault == "late_dropout":
                    pipe.send({"dropped_late": name, "sent": sent, "received": received,
                               "messages_out": messages_out, "messages_in": messages_in,
                               "wall_sec": time.perf_counter() - start})
                    return
                snapshot = recv()
                if snapshot.get("kind") != "uploads":
                    raise SecureAggregationError("Expected signed upload roster")
                proof = client.recover(announcements, snapshot["packets"], keys, ids, dim)
                send({"kind": "recovery", "packet": proof})
                answer = recv()
                if answer.get("kind") != "aggregate":
                    raise SecureAggregationError("Missing aggregate")
                pipe.send({"sum": answer["sum"], "receipt": answer["receipt"],
                           "sent": sent, "received": received,
                           "messages_out": messages_out, "messages_in": messages_in,
                           "wall_sec": time.perf_counter() - start,
                           "tls_version": sock.version()})
    except Exception as e:
        pipe.send({"error": f"{type(e).__name__}: {e}", "client": name,
                   "sent": sent, "received": received,
                   "messages_out": messages_out, "messages_in": messages_in})
    finally:
        pipe.close()


def tls_secure_sum_round(updates: Mapping[str, np.ndarray | list[float]],
                         *, missing=(), fault=None, timeout_sec=15.0, session=None):
    """Runs a complete multiprocess, mTLS loopback round.

    Returns (vector_sum, signed_crypto_receipt, metrics). The test controller
    holds input updates; only the worker processes separate those from the
    aggregation server. Fault = (client_name, 'late_dropout'|'tamper_upload').
    """
    names = sorted(updates)
    # Client names are also ephemeral certificate file basenames.
    # Reject slashes, path components, cert-name collisions and ambiguity.
    if any(not isinstance(name, str)
           or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name)
           or name.lower() == 'localhost' for name in names):
        raise ValueError('Client IDs must be safe, distinct certificate names')
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Two or more distinct clients required")
    if not set(missing).issubset(names):
        raise ValueError("Unknown missing client")
    if len(names) - len(missing) < 2:
        raise DropoutError("Fewer than two surviving clients")
    dim = len(np.asarray(updates[names[0]]))
    if not all(np.asarray(v).ndim == 1 and len(v) == dim for v in updates.values()):
        raise ValueError("All client vectors must have matching dimensions")
    if fault and (len(fault) != 2 or fault[0] not in names or fault[1] not in ("tamper_upload", "late_dropout")):
        raise ValueError("Invalid fault argument")
    start = time.perf_counter()
    session = session or secrets.token_hex(24)
    mp_ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="aquasmart_v45_tls_") as temp_dir:
        ca, credentials = _dev_pki(Path(temp_dir), names)
        procs = []
        pipes = []
        server_rx, server_tx = mp_ctx.Pipe(duplex=False)
        server = mp_ctx.Process(
            target=_server_worker,
            args=(server_tx, ca, *credentials["localhost"], names, dim, session, timeout_sec),
        )
        try:
            server.start()
            server_tx.close()
            if not server_rx.poll(timeout_sec):
                raise TimeoutError("TLS server did not bind before deadline")
            hello = server_rx.recv()
            if "error" in hello:
                raise SecureAggregationError(hello["error"])
            port = int(hello["port"])
            for name in names:
                rx, tx = mp_ctx.Pipe(duplex=False)
                proc = mp_ctx.Process(
                    target=_client_worker,
                    args=(tx, name, np.asarray(updates[name], dtype=np.float64).tolist(),
                          name in missing, ca, *credentials[name], port,
                          names, dim, session, timeout_sec,
                          fault[1] if fault and fault[0] == name else None),
                )
                proc.start()
                tx.close()
                procs.append(proc)
                pipes.append((name, rx))
            deadline = time.monotonic() + timeout_sec * 5
            client_reports = {}
            for name, rx in pipes:
                remains = max(0, deadline - time.monotonic())
                if not rx.poll(remains):
                    raise TimeoutError("Client " + name + " did not complete before deadline")
                client_reports[name] = rx.recv()
            remains = max(0, deadline - time.monotonic())
            if not server_rx.poll(remains):
                raise TimeoutError("Server did not complete before deadline")
            server_report = server_rx.recv()
            if "error" in server_report or any("error" in x for x in client_reports.values()):
                problems = [server_report.get("error", "")]
                problems.extend(x.get("error", "") for x in client_reports.values())
                raise DropoutError("Network round aborted: " + "; ".join(p for p in problems if p))
            if "aggregate" not in server_report:
                raise DropoutError("Server did not return an aggregate")
            survivors = [x for x in client_reports.values() if "sum" in x]
            if len(survivors) != len(names) - len(missing):
                raise DropoutError("Not all surviving clients received the result")
            answer = np.asarray(server_report["aggregate"], dtype=np.float64)
            if any(not np.allclose(answer, report["sum"], rtol=0, atol=1e-12)
                   for report in survivors):
                raise SecureAggregationError("Inconsistent aggregate received by survivors")
            metrics = {
                "n_clients": len(names), "n_survivors": len(survivors),
                "n_dropped": len(missing),
                "tls_application_frame_bytes": int(sum(x.get("sent", 0) + x.get("received", 0)
                                                        for x in client_reports.values())),
                "tls_application_frames": int(sum(x.get("messages_out", 0) + x.get("messages_in", 0)
                                                  for x in client_reports.values())),
                "network_round_wall_sec": float(time.perf_counter() - start),
                "client_max_wall_sec": float(max(x.get("wall_sec", 0) for x in client_reports.values())),
                "tls_protocol": "TLSv1.3 (mutual certificates, temporary test CA)",
                "measurement_scope": "application frames on loopback TCP; excludes TLS handshake and TCP/IP overhead",
            }
            return answer, server_report["receipt"], metrics
        finally:
            for p in procs:
                p.join(timeout=0.6)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            server.join(timeout=0.6)
            if server.is_alive():
                server.terminate()
                server.join(timeout=2)
            for _, rx in pipes:
                rx.close()
            server_rx.close()