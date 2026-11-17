AquaSmart V4.2 — Personalized Federated Learning (exploratory)
==============================================================

Dependencies (MUST already be installed in project):
- fl_smpc/temporal_protocol.py (V2)
- fl_smpc/run_temporal_experiments_v3.py (V3)
- fl_smpc/run_temporal_experiments_v4.py (V4.1)
- package requirements (numpy/pandas/scikit-learn)

This zip adds ONLY:
  fl_smpc/run_temporal_experiments_v42.py
  tests/test_temporal_v42.py
It DOES NOT replace V2/V3/V4.1, data.py, or the results.

1) Extract the zip into the PROJECT ROOT, merging its 'fl_smpc' and 'tests'
   folders with existing ones. Do not overwrite/replace the folders.

2) Verify/activate:
   cd /d C:\Users\dell\AquaSmart\AquaSmart-v2-FL-SMPC
   .venv\Scripts\activate
   python -m py_compile fl_smpc\run_temporal_experiments_v42.py
   python -m unittest discover -s tests -p "test_temporal_v42.py" -v

3) Single-seed pilot (selection on validation only):
   python -m fl_smpc.run_temporal_experiments_v42 --sample-cap 3000 --rounds 5,10,20 --lr-grid 0.005,0.01 --alpha-grid 0.1,1 --epochs-grid 1,5 --personal-lr-grid 0.001,0.005,0.01 --personal-epochs-grid 0,1,3,5 --seeds 42 --selection macro_rmse

4) After inspection, repeat with seeds 11 through 20:
   python -m fl_smpc.run_temporal_experiments_v42 --sample-cap 3000 --rounds 5,10,20 --lr-grid 0.005,0.01 --alpha-grid 0.1,1 --epochs-grid 1,5 --personal-lr-grid 0.001,0.005,0.01 --personal-epochs-grid 0,1,3,5 --seeds 11,12,13,14,15,16,17,18,19,20 --selection macro_rmse

5) Outputs: fl_smpc/results/temporal_v42/ (per-seed JSON and a summary JSON).

What this experiment does
-------------------------
1) Select FedAvg learning-rate, alpha, local epochs and rounds on VAL.
2) Start each pond-specific model at the selected global FedAvg weights.
3) Perform pond-local fine-tuning using ONLY local TRAIN samples, with common
   personalization epoch count and learning rate selected on aggregate VAL.
   Epochs=0 is available, making an exact global-model fallback.
4) Replay pairwise additive masks on global FedAvg and apply the same
   locally selected personalization. No additional client->server updates
   are transmitted during personalization.
5) Only after all choices are fixed, evaluate the same test data.

Important methodological limitations
------------------------------------
* feed_ration_g labels are synthetic/proxy, NOT measured intake or optimal ration.
* Pairwise additive masking is a single-process numerical emulator,
  NOT cryptographic SMPC, private key establishment, or dropout recovery.
* The same temporal test windows have already been examined in V2/V3/V4.1.
  Results remain exploratory, NOT independent final validation.
* Independent seed repeats vary training randomness on the same test windows.
* Tiny validation sets (notably ponds 10/11) risk instability in macro selection.
* The local/centralized/federated compute budgets are not matched.
* Validation metrics are computed centrally in a simulation. A real FL
  deployment would require a specified protocol for transmitting only the
  necessary per-pond validation aggregates.
* No guarantee that personalization improves performance over Local ML.
