"""Validate and clean the raw export."""

from pathlib import Path

import pandas as pd
import yaml

from src.rdd import validate

OUT = Path("data/processed/bank_data.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return validate(df).dropna()


def main() -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    raw = pd.read_csv(params["data"]["raw_path"])
    df = clean(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, lineterminator="\n")
    print(f"{len(raw)} rows in, {len(df)} rows out -> {OUT}")


if __name__ == "__main__":
    main()
