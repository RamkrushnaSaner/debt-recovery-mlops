"""Fit the full-data OLS model the API uses to predict recovery amounts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rdd import add_indicator, design_matrix, fit_ols


def main() -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    threshold = params["policy"]["threshold"]

    df = add_indicator(pd.read_csv("data/processed/bank_data.csv"), threshold)
    model = fit_ols(df)

    residuals = df["actual_recovery_amount"] - model.predict(design_matrix(df))
    metrics = {
        "r_squared": float(model.rsquared),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "coef_indicator": float(model.params["indicator"]),
    }

    Path("models").mkdir(exist_ok=True)
    model.save("models/ols_model.pkl", remove_data=True)  # drop training data from the file
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/train.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
