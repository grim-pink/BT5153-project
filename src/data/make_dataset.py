from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.clean_text import light_clean_text


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["v2"] if "v2" in df.columns else pd.Series([""] * len(df)),
        df["Unnamed: 2"] if "Unnamed: 2" in df.columns else pd.Series([""] * len(df)),
        df["Unnamed: 3"] if "Unnamed: 3" in df.columns else pd.Series([""] * len(df)),
        df["Unnamed: 4"] if "Unnamed: 4" in df.columns else pd.Series([""] * len(df)),
        df["text_type"] if "text_type" in df.columns else pd.Series([""] * len(df)),
        df["text"] if "text" in df.columns else pd.Series([""] * len(df)),
    ]
    combined = parts[0].fillna("").astype(str)
    for p in parts[1:]:
        combined = combined + " " + p.fillna("").astype(str)
    return combined


def make_dataset(spam_csv: str, dataset_csv: str, output_path: str) -> None:
    df1 = pd.read_csv(spam_csv, encoding="latin1")
    df2 = pd.read_csv(dataset_csv, encoding="latin1")

    df = pd.concat([df1, df2], ignore_index=True)
    df["v2"] = build_combined_text(df)

    drop_cols = [c for c in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "text_type", "text"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.rename(columns={"v1": "label", "v2": "text"})
    df["cleaned_text"] = df["text"].apply(light_clean_text)
    df = df.drop_duplicates(subset=["label", "cleaned_text"]).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved cleaned dataset to {output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spam_csv", required=True)
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    make_dataset(
        spam_csv=args.spam_csv,
        dataset_csv=args.dataset_csv,
        output_path=args.output_path,
    )
