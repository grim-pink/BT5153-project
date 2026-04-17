from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from src.intent.classify_intent import classify_intent
from src.intent.prompts import LABELS


def run_evaluation(df, text_col, label_col, mode: str):
    preds = []

    for text in df[text_col]:
        pred = classify_intent(text, mode=mode)
        preds.append(pred)

    report = classification_report(
        df[label_col],
        preds,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = f1_score(df[label_col], preds, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(df[label_col], preds, labels=LABELS, average="weighted", zero_division=0)
    cm = confusion_matrix(df[label_col], preds, labels=LABELS)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": preds,
    }


def evaluate(input_path, output_dir, text_col, label_col):
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)

    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    print(f"Loaded {len(df)} samples")

    print("\nRunning ZERO-SHOT evaluation...")
    zero_results = run_evaluation(df, text_col, label_col, mode="zero")

    print("\nRunning FEW-SHOT evaluation...")
    few_results = run_evaluation(df, text_col, label_col, mode="few")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_path / "zero_shot_metrics.json", "w") as f:
        json.dump(zero_results, f, indent=2)

    with open(output_path / "few_shot_metrics.json", "w") as f:
        json.dump(few_results, f, indent=2)

    # Save predictions
    df["zero_pred"] = zero_results["predictions"]
    df["few_pred"] = few_results["predictions"]
    df.to_csv(output_path / "comparison_predictions.csv", index=False)

    print("\n--- Results ---")
    print(f"Zero-shot Macro F1: {zero_results['macro_f1']:.4f}")
    print(f"Few-shot Macro F1:  {few_results['macro_f1']:.4f}")

    print("\nSaved outputs to:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default="artifacts/metrics")
    parser.add_argument("--text_col", default="cleaned_text")
    parser.add_argument("--label_col", default="Classification")

    args = parser.parse_args()

    evaluate(
        input_path=args.input_path,
        output_dir=args.output_dir,
        text_col=args.text_col,
        label_col=args.label_col,
    )