from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.clean_text import light_clean_text


def load_spam_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")

    extra_cols = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
    for col in extra_cols:
        if col not in df.columns:
            df[col] = ""

    df["text"] = (
        df["v2"].fillna("").astype(str) + " " +
        df["Unnamed: 2"].fillna("").astype(str) + " " +
        df["Unnamed: 3"].fillna("").astype(str) + " " +
        df["Unnamed: 4"].fillna("").astype(str)
    )

    df = df.rename(columns={"v1": "label"})
    return df[["label", "text"]]


def load_dataset_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df = df.rename(columns={"text_type": "label", "text": "text"})
    return df[["label", "text"]]


def make_dataset(spam_csv: str, dataset_csv: str, output_path: str) -> None:
    df1 = load_spam_csv(spam_csv)
    df2 = load_dataset_csv(dataset_csv)

    df = pd.concat([df1, df2], ignore_index=True)

    df["text"] = df["text"].fillna("").astype(str)
    df["cleaned_text"] = df["text"].apply(light_clean_text)

    df = df.drop_duplicates(subset=["label", "cleaned_text"]).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved cleaned dataset to {output}")
    print(f"Rows: {len(df)}")
    print(df["label"].value_counts())


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
