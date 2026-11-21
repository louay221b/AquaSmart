# SecFed — reproducible experiments (V1–V9)

This directory holds the **experiment code and unit tests** from the SecFed research review; it was imported additively and does not overwrite the separate AquaSmart `fl_smpc` experiments already committed on `research/improve-experiments`.

## Provenance and organization

- `sims/`: original exploratory experiment code and baselines (legacy).
- `correction_v1/`: aquaculture audit and chronological preprocessing.
- `correction_v2/`: federated local SGD personalization.
- `correction_v3/`: local and proximal adaptive variants.
- `correction_v4/`: freeze/test manifest and retrospective evaluation.
- `correction_v5/`: synchronized temporal partitions for aquaculture.
- `correction_v6/`: BDG2 audit, calendar validation, and experiments.
- `correction_v7/`: TCP/mTLS secure aggregation pilot and full recursive training, with source vendored from AquaSmart V4.5 for reproducibility.
- `correction_v8/`: paired cost benchmark (FedAvg over mTLS vs. secure aggregation).
- `correction_v9/`: targeted security-failure scenarios.
- `scripts/`: reproducible figure/table-generation scripts (require result JSONs exported separately; the six-panel generator expects the documented `scripts/data/` structure).

## Requirements

Python 3.12 is the tested runtime. Use the existing repository virtual environment or install the listed packages:

```bash
python -m pip install -r requirements.txt
```

From **this** `aina_pkg` directory, run the tests:

```bash
python -m unittest discover -s correction_v1/tests -v
python -m unittest discover -s correction_v2/tests -v
python -m unittest discover -s correction_v3/tests -v
python -m unittest discover -s correction_v4/tests -v
python -m unittest discover -s correction_v5/tests -v
python -m unittest discover -s correction_v6/tests -v
python -m unittest discover -s correction_v7/tests -v
python -m unittest discover -s correction_v8/tests -v
python -m unittest discover -s correction_v9/tests -v
```

Example workflows (run locally with separately downloaded datasets):

```bash
python correction_v5/run_synchronized.py --help
python correction_v6/audit_bdg2.py --help
python correction_v6/run_bdg2.py --help
python correction_v7/run_crypto_pilot.py --help
python correction_v7/run_crypto_full.py --help
python correction_v8/run_cost_benchmark.py --help
python correction_v9/run_security_validation.py --help
```

## Data and evaluation integrity

- Aquaculture CSVs are expected in the local project `dataset/` directory, **not committed**.
- Building Data Genome 2 (`metadata.csv`, `weather.csv`, `electricity.csv`) is expected in a local `bdg2_dataset/` folder. Check its redistribution license before sharing copies.
- V5/V6 frozen manifests and test JSON files are available in previously exported archives; these files are omitted from this *code-only* commit.
- Keep retrospective test acknowledgement switches explicit; do not retune after inspecting test values.
- The V7 mTLS prototype runs local loopback processes; its controller has access to plaintext contributions during orchestration. Do not equate its threat model with malicious-server secure aggregation.
- V8 reports application-level bytes and sampled summed process RSS rather than complete wire traffic or device memory.

## What is intentionally NOT committed

Datasets; training/test outputs; PDF manuscripts; private keys/certificates; runtime environments; temporary logs; generated figures; personal paths.

For ongoing work, see each `correction_v*/README*.md` and source-level CLI help.
