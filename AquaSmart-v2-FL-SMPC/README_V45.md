# AquaSmart V4.5 — TCP/mTLS network secure aggregation benchmark

**Research prototype / validation only.** This is an *additive* upgrade to V4.4.
It does not replace or edit older training scripts or datasets. Network secure
aggregation is based on V4.4's X25519+Ed25519+HKDF/ChaCha20 pairwise masking,
with support for signed recovery of clients who abandon **after key announcement
but before upload**.

## Files

- `fl_smpc/tls_secagg_v45.py`: true loopback TCP sockets with **TLS 1.3 mutual certificate authentication**, signed X25519 announcements, masked upload, recovery, aggregate, multiprocess clients/server, framing metrics.
- `fl_smpc/run_temporal_experiments_v45.py`: run on AquaSmart V2 chronological TRAIN/VAL splits. Use fixed hyperparameters from exploratory V4.2; **no TEST scoring**.
- `tests/test_temporal_v45.py`: eight tests (network success, dropout, corrupted signature, late dropout fail-closed, invalid rosters, and temporal integration).
- `SECURITY_MODEL_V45.md`: precise privacy and measurement limitations.

### Dependencies

Requires already-installed V2, V3, V4.1, V4.2, V4.3 and V4.4 Python scripts
and `fl_smpc/requirements.txt`, plus `requirements_v43.txt` (`cryptography`).
No new external dependency beyond those in V4.3.

The new code automatically generates a temporary TEST certificate authority,
server/client certificates and keys and deletes them at end of each run. These
certificates are **not** suitable for production. The server binds localhost
(`127.0.0.1`) only and each client authenticates with a separate certificate.

## Windows CMD installation

If you already ran `git pull` for the research branch and have local changes,
consider `git status` before pulling. Safest option for this incremental archive:

```cmd
cd /d C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC
.venv\Scripts\activate
powershell -NoProfile -Command "Expand-Archive -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\AquaSmart_V4_5_TLS_Network_SecAgg.zip') -DestinationPath (Get-Location).Path -Force"
python -m pip install -r requirements_v43.txt
python -m py_compile fl_smpc\tls_secagg_v45.py fl_smpc\run_temporal_experiments_v45.py
python -m unittest discover -s tests -p "test_temporal_v45.py" -v
```

Expected: `Ran 8 tests ... OK`. Tested in the author's Linux sandbox with
Python 3.11+/cryptography 46; Windows 3.12 must be verified locally.

## First real AquaSmart benchmark (11 ponds)

```cmd
python -m fl_smpc.run_temporal_experiments_v45 --sample-cap 3000 --rounds 2 --seeds 42 --dropouts IoTPond2,IoTPond7 --timeout-sec 45
```

Compares `all_online` vs `after_announcement_dropout` on the same validation
splits. Once successful, repeat a fixed set of seeds, e.g.:

```cmd
python -m fl_smpc.run_temporal_experiments_v45 --sample-cap 3000 --rounds 2 --seeds 11,12,13,14,15,16,17,18,19,20 --dropouts IoTPond2,IoTPond7 --timeout-sec 45
```

Outputs: `fl_smpc/results/temporal_v45/seed*.json` and
`summary_*.json` with validation RMSE, max FedAvg-vs-cryptographic deviation,
counts, actual *framed TLS application bytes*, and network round wall-clock
seconds (includes test cert setup, process startup, loopback and crypto).

## What is (and is not) measured

- **Yes:** Data is sent across real TCP sockets over authenticated TLS 1.3,
  among separately spawned OS processes on **one host**.
- **Yes:** JSON frame sizes crossing client TLS socket APIs are recorded
  (both directions); the TLS handshake is executed but **not captured in byte
  totals**. No packet capture, Ethernet/IP/TCP/TLS record overhead or WAN
  latency are included.
- **Yes:** The server process only receives signed key announcements, signed
  masked updates and signed dropout corrections, plus the aggregate. It does
  not receive each client's clear update as a protocol message.
- **No:** The benchmark parent **already has every client's clear update**:
  it calculates local SGD centrally and passes each weighted update to its
  separate client process at spawn. It is NOT private from that parent, and
  client data is NOT truly independent or remote yet.
- **No:** No defense against a dishonest coordinator inventing the dropout
  roster, a malicious client, client/server collusion, or prediction leakage
  from aggregate outputs. A late dropout **aborts the round safely**, rather
  than being recovered.
- **No:** `feed_ration_g` is a generated heuristic label. No untouched final
  TEST windows are consumed, and the fixed validation windows have been
  inspected previously. Do not present as deployment or real feeding gains.

## Next step

For an independent scientific network benchmark, replace the parent trainer
with a client-local loader/trainer on distinct hosts, provision a real CA and
trusted Ed25519 identities, and measure TCP/TLS packet wire bytes and WAN
latency under reproducible adversarial fault tests.