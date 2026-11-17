# AquaSmart V4.3 — Security model and reproducibility

**Status:** experimental cryptographic secure-sum prototype in one Python process. Not a complete, audited or deployed SMPC system.

## Participants and trust

Each simulated pond is a client with a private Ed25519 signing key and one newly generated X25519 ephemeral private key per training round. The aggregation server starts with an out-of-band trusted registry of client Ed25519 public verification keys. The server is *honest but curious* and does not collude with the clients. All participating clients are honest and remain online until completion. At least two non-colluding clients are required. There is no client anonymity.

The experiment orchestrator runs clients and the server in the same Python process. It has access to the training data and cannot be considered an isolated or adversarial server. Confidentiality of individual updates is a **property of the modeled protocol message interface under the assumptions**, not a proven deployment guarantee.

## Protocol steps

For every global FedAvg round:

1. Create an unpredictable round ID. Each client generates a fresh X25519 ephemeral keypair and signs `(protocol-version, round-id, client-id, public-key)` using its Ed25519 identity key.
2. Broadcast the same full roster of signed announcements to all clients; each independently checks the signatures against the pretrusted registry, exact participant set, and its own announced key. The canonical transcript includes all signed announcements and integer encoding parameters.
3. Locally train the client's weighted model update `u_i = n_i / N * theta_i` using V4.2's SGD configuration.
4. Quantize each coordinate with scale `s = 10^8`, checking a per-coordinate bound `|u_ij| <= 10,000`, and map to the field modulo `p = 2^61 - 1`. Configuration checks ensure an honest aggregate cannot wrap around the signed field representation.
5. For each unordered client pair `(i,j)`, derive the same X25519 shared secret. HKDF-SHA256 with a transcript-bound salt and pair-bound info produces a unique ChaCha20 key. Run ChaCha20 once per pair and round to produce the masking field vector. Lower-ID client adds its mask, higher-ID subtracts it.
6. Each client signs the complete masked vector, its round and transcript digest, and uploads it. Server verifies exact roster membership, signatures, round/transcript and vector dimensions.
7. Only when all expected messages are present, sum the masked integer vectors modulo `p`; pairwise masks cancel. Decode the signed fixed-point sum to floating-point global parameters. If a client misses the round, **abort and return no aggregate**.
8. The plain global and secure global models are independently fine-tuned using the **same** V4.2 validation-selected local configuration; numeric discrepancies are measured without changing the selection.

Equations (coordinate-wise):

`q_i = round(s * u_i) mod p`

`z_i = q_i + sum_{j>i} m_ij - sum_{j<i} m_ji mod p`

`sum_i z_i mod p = sum_i q_i mod p`

The decoded integer sum differs from a direct floating-point sum by at most `n / (2*s)` per coordinate before normal floating-point/optimizer effects (provided all clients obey the range checks and modular bound).

## Guarantees / non-guarantees

- Cryptographic primitives protect the *pairwise masks* from an honest-but-curious server that only sees the signed announcements and masked updates, assuming non-collusion and accepted cryptographic assumptions.
- An Ed25519 signature does not make a malicious client's numerical update valid; there are no proofs of bounded inputs or Byzantine robust aggregation.
- An aggregate can itself leak information. There is **no differential privacy** and no guarantee against collusions, manipulated participant sets, or chosen-input attacks.
- No dropout recovery or Shamir secret sharing. Aborted rounds are safe but unavailable.
- No TCP/TLS, real clients, persistent PKI, key lifecycle, remote attestation, network fault injection, or attack-resistant key isolation.
- The secure server object receives only signed masked vectors, but the experiment's single-process Python orchestrator contains all client datasets. Do not market this as data-private distributed deployment.
- Protocol timing measures *in-memory operations*, and bytes represent JSON-serialized messages plus an estimated raw float model broadcast. Network latency and transport overhead have not been measured.
- Empirical model accuracy uses heuristic synthetic feed labels on previously inspected temporal test windows; independent data and time windows are needed before publication claims.

## Reproduction

```
python -m pip install -r requirements_v43.txt
python -m unittest discover -s tests -p "test_temporal_v43.py" -v
python -m fl_smpc.run_temporal_experiments_v43 --sample-cap 3000 --rounds 5,10,20 --lr-grid 0.005,0.01 --alpha-grid 0.1,1 --epochs-grid 1,5 --personal-lr-grid 0.001,0.005,0.01 --personal-epochs-grid 0,1,3,5 --seeds 42 --selection macro_rmse
```

Only the two new `fl_smpc/` files, unit tests, and this documentation are included in this release; existing V2–V4.2 code is not overwritten.
