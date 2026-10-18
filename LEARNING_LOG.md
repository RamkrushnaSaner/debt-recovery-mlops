## 2026-10-18 — Day 2: RDD analysis as code, MLflow tracking, model registry

**Three things I can now explain without notes**

1. **Experiment tracking.** Every `python -m src.evaluate` run is logged to MLflow on DagsHub with its params, 18 metrics, the model files and a `data_md5` tag, so every number traces back to the exact settings and data. Rerunning threshold 1000 / bandwidth 150 gave identical numbers every time (211 customers, jump 316.75, ROI 1.553).
2. **Model registry.** Registering a run's model gives it a name and a version (`recovery-ols` v1). The alias `@champion` lets any code load the approved model without knowing the run id: `models:/recovery-ols@champion`.
3. **Placebo test.** At a fake threshold of 1500, inside Level 1 where the strategy does not change, the jump is 58 (p = 0.67, 95% CI −210 to 327, includes 0). At the real threshold of 1000 it is 317 (p = 0.000008, CI 180 to 453). No jump where nothing changes makes the $1000 effect more believable. The raw uplift was still +117 at 1500 because larger debts repay more anyway, which is why the regression coefficient decides, not the raw difference. Caveat: the placebo interval is wide, so it shows no evidence of a jump rather than proving there is none.

**Decision:** keep bandwidth 150. At 100 the age balance is borderline (p = 0.054); wider windows (200, 300) raise the uplift-based ROI mainly because they take in more of the natural slope, while the regression jump shrinks (349 → 317 → 282 → 270).

**One error and its fix**

- `register_model` failed with "Unable to find a logged_model with artifact_path model". MLflow 3's `log_model` stores the model as a separate object, and DagsHub does not link it back to the run. Fix: `save_model` into a temporary folder, then `log_artifacts(..., artifact_path="model")`, so the files live inside the run (commit 7c1a5b9).
