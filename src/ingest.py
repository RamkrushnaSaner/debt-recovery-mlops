"""Import the bank's export into data/raw after validating its schema.

--frac below 1.0 simulates an earlier, smaller export (dataset v1).
"""

import argparse
from pathlib import Path

import pandas as pd

from src.rdd import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path to the bank's bank_data.csv")
    parser.add_argument("--out", default="data/raw/bank_data.csv")
    parser.add_argument("--frac", type=float, default=1.0, help="share of customers to keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = validate(pd.read_csv(args.source))
    if args.frac < 1.0:
        df = df.sample(frac=args.frac, random_state=args.seed).sort_index()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {args.out}")


if __name__ == "__main__":
    main()
