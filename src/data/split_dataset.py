from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_path: str,
    train_output: str,
    test_output: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    df = pd.read_csv(input_path)

    # ── 1. Validate required columns ─────────────────────────────
    required_cols = {"label", "text", "cleaned_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── 2. Normalize labels ──────────────────────────────────────
    df["label"] = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Keep only valid labels
    df = df[df["label"].isin(["ham", "spam"])].copy()

    # ── 3. Create numeric label ──────────────────────────────────
    df["label_num"] = (df["label"] == "spam").astype(int)

    # ── 4. Train-test split (stratified) ─────────────────────────
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_num"],
    )

    # ── 5. Save outputs ──────────────────────────────────────────
    train_out = Path(train_output)
    test_out = Path(test_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    # ── 6. Debug prints (VERY useful) ────────────────────────────
    print(f"Train saved to {train_out} with {len(train_df)} rows")
    print(f"Test saved to {test_out} with {len(test_df)} rows")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nTest label distribution:")
    print(test_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--test_output", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        input_path=args.input_path,
        train_output=args.train_output,
        test_output=args.test_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
