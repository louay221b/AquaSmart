# AquaSmart V4.4 — Security model and precise limits

**Status:** Research prototype, not audited, not safe to deploy as-is.

## What is new compared with V4.3?

V4.3 required all announced clients to upload. V4.4 permits a client that
**signed and sent its ephemeral public-key announcement, but did not upload
any masked update**, to disappear. The protocol computes the sum of the
updates actually uploaded by the remaining clients, provided **at least two
clients uploaded** and **all uploaders remain online through recovery**.

## Workflow

1. Each active client announces a fresh X25519 ephemeral public key signed
   using Ed25519. All participants use the same authenticated roster.
2. Clients locally compute fixed-point weighted model updates and hide them
   under additive pairwise ChaCha20 masks derived via X25519/HKDF-SHA256.
3. The coordinator collects signed masked uploads. Some announced clients may
   be absent at this point. The upload set is frozen.
4. Each surviving uploader checks that its signed upload is included and
   validates the signatures of *all* listed uploads, then sends a signed
   correction that removes only masks it shares with missing non-uploaders.
   **No surviving-to-surviving masks are disclosed.**
5. The server verifies all signed uploads and corrections, rejects incomplete
   recovery, and returns the plaintext **sum of uploaded client updates**.
   The FedAvg experiment divides this sum by the sum of survivor weights.
6. Ephemeral DH private keys are discarded after the recovery step.

## Threat and privacy model

- **Honest clients; honest-but-curious coordinator.** The coordinator follows
  protocol faithfully, does not falsely omit received uploads, does not
  collude with clients, and sees signed protocol packets. It does not receive
  plaintext individual weighted updates at its *process interface*.
- Two or more uploaders must survive. Fewer than two → abort, not a result.
- Clients' identity public keys must be authenticated/provisioned independently
  in an actual deployment. The laboratory launcher self-registers the clients'
  freshly generated public keys, which is not production identity management.
- Participants are trusted to follow the protocol. An actively cheating
  client can send an incorrect correction, which is signed but not
  cryptographically proven correct. V4.4 does not prevent malicious updates.
- An actively dishonest coordinator could falsely claim a client has dropped
  and cause revealing of pairwise masks. **No malicious-server security is
  claimed.** This is NOT a full Bonawitz-style threshold secret-sharing secure
  aggregation and there is no Byzantine adversarial security theorem.
- As with V4.3, the aggregate itself may reveal information, especially for
  small cohorts; this does not provide differential privacy.

## Unsupported failures

- A client uploading a masked vector, then disappearing before sending a
  required recovery correction → secure round aborts.
- Compromised/lying server, colluding clients, malicious client updates,
  persistent authentication keys, and secure key rotation: not handled.
- Cross-round aggregation metadata privacy is not provided.

## Multiprocessing benchmark

`fl_smpc/process_secagg_v44.py` uses `spawn` to isolate each client's signing
and ephemeral DH private keys in a separate OS process. A separate aggregator
process receives **only protocol messages and public keys**.

The experiment parent orchestrates and holds plaintext input vectors before
process creation; therefore it is **not a deployed private training system**.
The parent measures JSON bytes passed via local `multiprocessing.Pipe`, **not**
network bytes, TLS traffic or socket overhead. The `process_secure_sum()`
function's `ipc_payload_bytes` excludes the plaintext vectors serialized as
process-start arguments and operating-system process initialization traffic.

The benchmark reports wall time including process spawn, userland crypto and
IPC; it must not be interpreted as standalone cryptographic CPU overhead.

## Scientific reproducibility

`run_temporal_experiments_v44.py` uses only TRAIN and VALIDATION; no TEST read.
The hyperparameters are *fixed*, not retuned: FedAvg lr=0.01, alpha=0.1,
local_epochs=5; personalized lr=0.01, epochs=3. Other candidate settings can
be supplied, but this tool is not a hyperparameter search.

Dropout alters the population of contributing clients; it necessarily changes
FedAvg and potentially the prediction error. Evaluation on all 11 ponds (incl.
the dropped client) is reported as *validation* only.

The label `feed_ration_g` is derived from a heuristic, not observed feeding
amounts. Previously inspected test windows remain exploratory.

## Next implementation phase

For stronger dropout security, incorporate cryptographic threshold secret
sharing (with a formally specified malicious-server threat model) and secure
transport among independently deployed clients and server. Add authentication,
timeouts, rekeying and a repeatable TCP/TLS network benchmark, then perform a
new truly independent temporal evaluation.