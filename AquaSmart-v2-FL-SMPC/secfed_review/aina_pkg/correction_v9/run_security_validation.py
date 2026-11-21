"""SecFed V9A - controlled fail-closed tests of the unchanged V7 TLS protocol.

This is an adversarial *test harness*, NOT a cryptographic security proof.
Only synthetic updates are used; no aquaculture/BDG2 files are accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import ssl
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'correction_v7' / 'vendor'
if not (VENDOR / 'fl_smpc' / 'tls_secagg_v45.py').exists():
    raise RuntimeError('V7A is missing: correction_v7/vendor/fl_smpc/tls_secagg_v45.py')
sys.path.insert(0, str(VENDOR))

from fl_smpc.tls_secagg_v45 import tls_secure_sum_round
from fl_smpc.secure_aggregation import SCALE, SecureAggregationError, DropoutError
from fl_smpc.secure_aggregation_v44 import (
    RecoverableClient, verify_roster, verify_uploads,
)


def inputs(seed: int, n: int, dim: int) -> dict[str, np.ndarray]:
    if n < 3 or dim < 1:
        raise ValueError('At least 3 clients and 1 dimension required')
    rng = np.random.default_rng(np.random.SeedSequence([2026, 9, seed, n, dim]))
    return {f'C{i:03d}': rng.uniform(-.5, .5, dim) / n for i in range(n)}


def _trial(name: str, action, *, expect_rejection: bool = False,
           require_signature_diagnostic: bool = False) -> dict:
    start = time.perf_counter()
    try:
        details = action()
    except SecureAggregationError as exc:
        diagnostic = str(exc)
        correct_type = (not require_signature_diagnostic or
                        'Invalid authenticated client signature' in diagnostic)
        return {
            'scenario': name, 'passed': bool(expect_rejection and correct_type),
            'observed': 'rejected', 'exception_type': type(exc).__name__,
            'error_message': diagnostic[:1500],
            'signature_failure_detected': ('Invalid authenticated client signature' in diagnostic),
            'elapsed_sec': time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            'scenario': name, 'passed': False, 'observed': 'unexpected_exception',
            'exception_type': type(exc).__name__, 'error_message': str(exc)[:1500],
            'elapsed_sec': time.perf_counter() - start,
        }
    return {
        'scenario': name, 'passed': not expect_rejection,
        'observed': 'accepted', 'details': details,
        'elapsed_sec': time.perf_counter() - start,
    }


def _positive(updates, missing, timeout):
    got, receipt, met = tls_secure_sum_round(updates, missing=missing, timeout_sec=timeout)
    keep = [x for x in sorted(updates) if x not in missing]
    expected = np.array([updates[k] for k in keep]).sum(axis=0)
    delta = float(np.max(np.abs(got - expected)))
    bound = len(keep) / (2 * SCALE) + 1e-10
    if delta > bound:
        raise AssertionError('Fixed-point aggregate outside certified rounding bound')
    if receipt.get('n_uploaded') != len(keep) or receipt.get('n_dropped') != len(missing):
        raise AssertionError('Incorrect signed-roster accounting in receipt')
    if bool(receipt.get('recovery_used')) != bool(missing):
        raise AssertionError('Incorrect recovery flag')
    if not str(met.get('tls_protocol', '')).startswith('TLSv1.3'):
        raise AssertionError('TLS 1.3 not confirmed')
    return {
        'max_sum_error': delta, 'rounding_bound': bound,
        'uploaded_count': receipt['n_uploaded'], 'dropout_count': receipt['n_dropped'],
        'recovery_used': bool(receipt['recovery_used']),
        'tls_protocol': met['tls_protocol'],
        'application_frame_bytes': int(met['tls_application_frame_bytes']),
        'application_frame_count': int(met['tls_application_frames']),
        'network_round_wall_sec': float(met['network_round_wall_sec']),
    }


def _direct_signature_forgery(updates, field: str):
    names = sorted(updates)
    d = len(next(iter(updates.values())))
    session = 'v9a-signature-check-' + field
    clients = {k: RecoverableClient(k) for k in names}
    keys = {k: c.identity_public_key for k, c in clients.items()}
    roster = [clients[k].announce(session) for k in names]
    actual = [clients[k].upload(updates[k], roster, keys, names, d) for k in names]
    proof = [dict(m) for m in actual]
    if field == 'signature':
        proof[0]['signature'] = '00' * 64
    elif field == 'masked':
        proof[0]['masked'] = list(proof[0]['masked'])
        proof[0]['masked'][0] = (proof[0]['masked'][0] + 1) % ((1 << 61) - 1)
    else:
        raise ValueError('Unsupported forgery')
    # The original transcript remains valid, but its signed masked payload was forged.
    _, transcript = verify_roster(roster, keys, names, session, d)
    verify_uploads(proof, keys, names, transcript, session, d)
    return {'unexpected': 'forged upload accepted'}


def _replay_old_session(updates):
    names = sorted(updates)
    d = len(next(iter(updates.values())))
    clients = {k: RecoverableClient(k) for k in names}
    keys = {k: c.identity_public_key for k, c in clients.items()}
    old_session = 'v9a-old-session'
    new_session = 'v9a-new-session'
    old_roster = [clients[k].announce(old_session) for k in names]
    verify_roster(old_roster, keys, names, new_session, d)
    return {'unexpected': 'replayed announcements accepted'}


def suite(seed: int = 0, n: int = 3, dim: int = 8, timeout: float = 15) -> list[dict]:
    """Eight scenarios: success, recoverable dropout and targeted rejections."""
    updates = inputs(seed, n, dim)
    who = sorted(updates)[1]
    too_many_missing = tuple(sorted(updates)[:-1])
    records = [
        _trial('normal_mtls', lambda: _positive(updates, (), timeout)),
        _trial('dropout_before_upload', lambda: _positive(updates, (who,), timeout)),
        _trial('forged_signature_direct', lambda: _direct_signature_forgery(updates, 'signature'),
               expect_rejection=True, require_signature_diagnostic=True),
        _trial('modified_signed_payload_direct', lambda: _direct_signature_forgery(updates, 'masked'),
               expect_rejection=True, require_signature_diagnostic=True),
        _trial('stale_announcement_replay', lambda: _replay_old_session(updates),
               expect_rejection=True),
        _trial('forged_upload_over_mtls', lambda: tls_secure_sum_round(
            updates, fault=(who, 'tamper_upload'), timeout_sec=timeout),
               expect_rejection=True, require_signature_diagnostic=True),
        _trial('late_dropout_after_upload', lambda: tls_secure_sum_round(
            updates, fault=(who, 'late_dropout'), timeout_sec=timeout),
               expect_rejection=True),
        _trial('insufficient_survivors', lambda: tls_secure_sum_round(
            updates, missing=too_many_missing, timeout_sec=timeout),
               expect_rejection=True),
    ]
    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--clients', type=int, default=3)
    p.add_argument('--dimension', type=int, default=8)
    p.add_argument('--timeout', type=float, default=15)
    p.add_argument('--output', type=Path, default=Path('security_results/v9a_security.json'))
    args = p.parse_args()
    if args.repeats < 1 or args.clients < 3 or args.dimension < 1 or args.timeout <= 0:
        p.error('Invalid repeats/clients/dimension/timeout')
    if args.output.exists():
        p.error('Output exists. Choose an unused --output to preserve the original.')
    files = [Path(__file__), *sorted((VENDOR / 'fl_smpc').glob('*.py'))]
    result = {
        'protocol': 'secfed-v9a-adversarial-validation-1',
        'complete': False,
        'context': 'Synthetic, test-only contributions and test CA, loopback mTLS',
        'parameters': {'repeats': args.repeats, 'clients': args.clients,
                       'dimension': args.dimension, 'timeout_sec': args.timeout},
        'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                        'openssl': ssl.OPENSSL_VERSION, 'numpy': np.__version__},
        'code_sha256': {str(path.relative_to(ROOT)).replace(os.sep, '/'):
                        hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'limitations': [
            'Controller creates and knows every synthetic plaintext update.',
            'A test abort is not a proof of malicious-server, collusion or threshold security.',
            'Identity keys come from in-band announcements, not a durable pinned client registry.',
            'This prototype uses TLS loopback with a freshly generated test CA.',
            'Only pre-upload dropout recovery is supported; post-upload dropout aborts.',
            'Message volumes on successful rounds are application-frame bytes, not wire bytes.',
        ],
        'trials': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(args.repeats):
        for test in suite(rep, args.clients, args.dimension, args.timeout):
            result['trials'].append({'repeat': rep, **test})
            print(f"repeat={rep} {test['scenario']}: {test['observed']} "
                  f"({'PASS' if test['passed'] else 'FAIL'}) "
                  f"{test['elapsed_sec']:.3f}s", flush=True)
    result['complete'] = all(row['passed'] for row in result['trials'])
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(args.output)
    print(f"Passed {sum(x['passed'] for x in result['trials'])}/{len(result['trials'])} scenarios")
    print(f'Saved: {args.output}', flush=True)
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
