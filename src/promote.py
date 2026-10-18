"""Promote the newest run for a policy setting to recovery-ols@champion.

Safe to run again: a run that is already registered is not registered twice.
Usage: python -m src.promote --threshold 1000 --bandwidth 150
"""

import argparse

import mlflow
import yaml
from mlflow import MlflowClient

MODEL_NAME = "recovery-ols"
ALIAS = "champion"


def newest_run(experiment: str, threshold: int, bandwidth: int) -> str:
    """Return the id of the newest finished, non-placebo run for these settings."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"params.threshold = '{threshold}' and params.bandwidth = '{bandwidth}'",
        order_by=["start_time DESC"],
    )
    if not runs.empty:
        runs = runs[runs["status"] == "FINISHED"]
        if "tags.purpose" in runs:
            runs = runs[runs["tags.purpose"] != "placebo"]
    if runs.empty:
        raise SystemExit(
            f"No finished, non-placebo run for threshold={threshold}, bandwidth={bandwidth}"
        )
    return runs.iloc[0].run_id


def version_for_run(client: MlflowClient, run_id: str) -> str | None:
    """Return the registered version that came from this run, if there is one."""
    for mv in client.search_model_versions(f"name = '{MODEL_NAME}'"):
        if mv.run_id == run_id:
            return mv.version
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a run to the champion model.")
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--bandwidth", type=int, required=True)
    args = parser.parse_args()

    with open("params.yaml", encoding="utf-8") as f:
        experiment = yaml.safe_load(f)["mlflow"]["experiment_name"]

    run_id = newest_run(experiment, args.threshold, args.bandwidth)
    print(f"newest run for threshold={args.threshold}, bandwidth={args.bandwidth}: {run_id}")

    client = MlflowClient()
    version = version_for_run(client, run_id)
    if version is None:
        version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
        print(f"registered as {MODEL_NAME} version {version}")
    else:
        print(f"already registered as {MODEL_NAME} version {version}; not registering again")

    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(f"{MODEL_NAME}@{ALIAS} -> version {version}")


if __name__ == "__main__":
    main()
