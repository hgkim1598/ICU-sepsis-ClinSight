from __future__ import annotations

import argparse
from pathlib import Path

from model import build_ensemble_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb_path", type=str, required=True)
    parser.add_argument("--gru_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="weighted_ensemble_predictions.csv")
    parser.add_argument("--weight_xgb", type=float, default=0.5)
    parser.add_argument("--weight_gru", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = build_ensemble_dataframe(
        xgb_path=args.xgb_path,
        gru_path=args.gru_path,
        weight_xgb=args.weight_xgb,
        weight_gru=args.weight_gru,
        threshold=args.threshold,
    )

    output_path = Path(args.output_path)
    df.to_csv(output_path, index=False)

    print("Saved:")
    print(f"- {output_path}")


if __name__ == "__main__":
    main()