AquaSmart ML + FL + masked aggregation — Temporal protocol V3
==============================================================

WHAT CHANGED
- New file fl_smpc/run_temporal_experiments_v3.py. V2 and original scripts remain untouched.
- Same per-pond whole-calendar-day train/validation/test partitions for all methods.
- Candidate training budgets are evaluated on VALIDATION only, then a single
  chosen checkpoint is evaluated on TEST for each method.
- Centralized and Local select an epoch budget (round candidate * local_epochs);
  FedAvg selects a round budget. Each approach selects independently on validation.
- Masking emulator replays FedAvg at the PLAIN FedAvg selected round, with the
  same RNG schedule; it has no independent search on validation/test.
- Per-seed JSON results include validation history, selected checkpoint,
  TEST micro, macro and per-pond metrics, and analytical communication estimates.
- One aggregate JSON summarizes micro TEST RMSE mean/stdev over seeds.

REQUIRES
- The existing fl_smpc/temporal_protocol.py from V2.
- The original fl_smpc/data.py and fl_smpc/common.py.
- Dependencies already installed in your .venv.

INSTALL (Windows CMD, in AquaSmart-v2-FL-SMPC)
1. Extract the ZIP to a temporary folder.
2. Copy ONLY fl_smpc/run_temporal_experiments_v3.py into the existing project's fl_smpc/.
3. Optionally copy tests/test_temporal_v3.py into project's tests/ (create tests directory if missing).
4. cd /d C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC
5. .venv\Scripts\activate
6. python -m py_compile fl_smpc\run_temporal_experiments_v3.py
7. python -m unittest discover -s tests -p test_temporal_v3.py -v
8. python -m fl_smpc.run_temporal_experiments_v3 --sample-cap 3000 --rounds 5,10,20 --seeds 42
9. Once verified, more seeds:
   python -m fl_smpc.run_temporal_experiments_v3 --sample-cap 3000 --rounds 5,10,20,50 --seeds 11,12,13,14,15

SCIENTIFIC CAUTIONS
- Test metrics are NOT used for selecting hyperparameters. Avoid choosing a
  different setup because it has a better TEST score; that would defeat the holdout.
- Defaults only tune TRAINING BUDGET, not LR or regularization. For learning rate
  tuning, use validation with predeclared candidate grid and nested selection.
- These methods are not compute-matched: local and centralized see different
  examples per epoch than FedAvg. Report this and compare costs separately.
- IoTPond10/11 contain tiny validation subsets; IoTPond12 has few test days.
- Current target is simulated from sensor readings, NOT actual feed amounts.
- This emulator does not implement secure key agreement, secret sharing,
  client dropout recovery, collusion robustness, or real network traffic.
- Time is local sequential training time, not end-to-end distributed latency.
- Calendar-day strings are not timezone-normalized; verify units and local dates.
- Synthetic trials/test fixture exercise code paths; no scientific validity
  can be inferred without checking the real dataset and experimental outputs.
