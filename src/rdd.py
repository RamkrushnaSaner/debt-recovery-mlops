"""Regression discontinuity (RDD) logic as pure functions: no files, no MLflow."""

import pandas as pd
import statsmodels.api as sm
from scipy import stats

REQUIRED_COLUMNS = [
    "id",
    "expected_recovery_amount",
    "actual_recovery_amount",
    "recovery_strategy",
    "age",
    "sex",
]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if (df["expected_recovery_amount"] < 0).any():
        raise ValueError("expected_recovery_amount must be >= 0")
    return df


def add_indicator(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["indicator"] = (out["expected_recovery_amount"] >= threshold).astype(int)
    return out


def rdd_window(df: pd.DataFrame, threshold: float, bandwidth: float) -> pd.DataFrame:
    """Customers with threshold - bandwidth <= expected < threshold + bandwidth."""
    lower, upper = threshold - bandwidth, threshold + bandwidth
    mask = (df["expected_recovery_amount"] >= lower) & (df["expected_recovery_amount"] < upper)
    return add_indicator(df.loc[mask], threshold)


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """const + expected_recovery_amount + indicator, in a fixed order (train and serve)."""
    X = df[["expected_recovery_amount", "indicator"]].astype(float)
    return sm.add_constant(X, has_constant="add")


def fit_ols(df: pd.DataFrame):
    return sm.OLS(df["actual_recovery_amount"], design_matrix(df)).fit()


def evaluate_rdd(df: pd.DataFrame, threshold: float, bandwidth: float) -> dict:
    w = rdd_window(df, threshold, bandwidth)
    control, treatment = w[w["indicator"] == 0], w[w["indicator"] == 1]

    # Balance checks: customers just below and just above should look alike
    _, kw_age_p = stats.kruskal(control["age"], treatment["age"])
    _, chi2_sex_p, _, _ = stats.chi2_contingency(pd.crosstab(w["indicator"], w["sex"]))

    # Effect: does recovery jump at the threshold?
    _, kw_recovery_p = stats.kruskal(
        control["actual_recovery_amount"], treatment["actual_recovery_amount"]
    )
    model = fit_ols(w)
    ci = model.conf_int().loc["indicator"]
    mean_c = control["actual_recovery_amount"].mean()
    mean_t = treatment["actual_recovery_amount"].mean()
    return {
        "n_samples": len(w),
        "kw_age_pvalue": float(kw_age_p),
        "chi2_sex_pvalue": float(chi2_sex_p),
        "mean_recovery_control": float(mean_c),
        "mean_recovery_treatment": float(mean_t),
        "recovery_uplift": float(mean_t - mean_c),
        "kw_recovery_pvalue": float(kw_recovery_p),
        "coef_indicator": float(model.params["indicator"]),
        "p_value_indicator": float(model.pvalues["indicator"]),
        "ci_lower_95": float(ci.iloc[0]),
        "ci_upper_95": float(ci.iloc[1]),
        "r_squared": float(model.rsquared),
    }


def roi(recovery_uplift: float, extra_cost: float) -> dict:
    net_gain = recovery_uplift - extra_cost
    return {"net_gain": float(net_gain), "roi": float(net_gain / extra_cost)}
