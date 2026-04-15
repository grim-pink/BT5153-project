from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def train_baseline(
    train_path: str,
    test_path: str,
    model_output: str,
    metrics_output: str,
    experiment_name: str = "Spam_Classification_Baseline",
) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["cleaned_text"].astype(str).tolist()
    y_train = train_df["label_num"].astype(int).tolist()
    X_test = test_df["cleaned_text"].astype(str).tolist()
    y_test = test_df["label_num"].astype(int).tolist()

    model_output_path = Path(model_output)
    metrics_output_path = Path(metrics_output)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="TFIDF_LogisticRegression"):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "fpr": float(fpr),
            "cm_tn": int(tn),
            "cm_fp": int(fp),
            "cm_fn": int(fn),
            "cm_tp": int(tp),
        }

        print("--- Baseline Metrics ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"Confusion Matrix:\n[[{tn}, {fp}]\n [{fn}, {tp}]]")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metrics(metrics)

        joblib.dump(pipe, model_output_path)
        mlflow.sklearn.log_model(pipe, artifact_path="baseline_model")
        mlflow.log_artifact(str(model_output_path))

        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--model_output", default="artifacts/baseline/model.joblib")
    parser.add_argument("--metrics_output", default="artifacts/metrics/baseline_metrics.json")
    parser.add_argument("--experiment_name", default="Spam_Classification_Baseline")
    args = parser.parse_args()

    train_baseline(
        train_path=args.train_path,
        test_path=args.test_path,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        experiment_name=args.experiment_name,
    )
