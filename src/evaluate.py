"""Evaluate the policy (RDD + ROI) and log the whole experiment to MLflow."""

import hashlib
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import statsmodels.api as sm
import yaml

from src.rdd import evaluate_rdd, roi

DATA = "data/processed/bank_data.csv"


def main() -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    p = params["policy"]

    df = pd.read_csv(DATA)
    result = evaluate_rdd(df, p["threshold"], p["bandwidth"])
    result.update(roi(result["recovery_uplift"], p["extra_cost"]))
    policy = {**p, **result}

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8", newline="\n"
    )
    train_metrics = json.loads(Path("metrics/train.json").read_text(encoding="utf-8"))

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(p)
        mlflow.log_metrics(result)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        # Link this run to the exact data version DVC tracks
        mlflow.set_tag("data_md5", hashlib.md5(Path(DATA).read_bytes()).hexdigest())
        # Store the model files inside the run (works on any MLflow server, DagsHub included)
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.statsmodels.save_model(sm.load("models/ols_model.pkl"), f"{tmp}/model")
            mlflow.log_artifacts(f"{tmp}/model", artifact_path="model")
        mlflow.log_artifact("metrics/policy.json")

    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
