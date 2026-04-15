from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "cm_tn": int(tn),
        "cm_fp": int(fp),
        "cm_fn": int(fn),
        "cm_tp": int(tp),
    }


def train_distilbert(
    train_path: str,
    test_path: str,
    model_output_dir: str,
    metrics_output: str,
    experiment_name: str = "Spam_Classification_DistilBERT",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 64,
    learning_rate: float = 2e-5,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_train_epochs: int = 2,
    weight_decay: float = 0.01,
) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["cleaned_text"].astype(str).tolist()
    y_train = train_df["label_num"].astype(int).tolist()
    X_test = test_df["cleaned_text"].astype(str).tolist()
    y_test = test_df["label_num"].astype(int).tolist()

    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
    test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_output_path = Path(model_output_dir)
    metrics_output_path = Path(metrics_output)
    model_output_path.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="DistilBERT_Finetune"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("eval_batch_size", eval_batch_size)
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("fp16", torch.cuda.is_available())

        training_args = TrainingArguments(
            output_dir=str(model_output_path / "checkpoints"),
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="mlflow",
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("🚀 Starting DistilBERT fine-tuning...")
        trainer.train()

        results = trainer.evaluate()

        metrics = {
            "precision": float(results["eval_precision"]),
            "recall": float(results["eval_recall"]),
            "f1_score": float(results["eval_f1"]),
            "fpr": float(results["eval_fpr"]),
            "cm_tn": int(results["eval_cm_tn"]),
            "cm_fp": int(results["eval_cm_fp"]),
            "cm_fn": int(results["eval_cm_fn"]),
            "cm_tp": int(results["eval_cm_tp"]),
        }

        print("\n--- DistilBERT Optimized Metrics ---")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
        print(
            f"Confusion Matrix:\n[[{metrics['cm_tn']}, {metrics['cm_fp']}]\n "
            f"[{metrics['cm_fn']}, {metrics['cm_tp']}]]"
        )

        trainer.save_model(str(model_output_path))
        tokenizer.save_pretrained(str(model_output_path))

        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_output_path))
        mlflow.log_artifacts(str(model_output_path), artifact_path="distilbert_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--model_output_dir", default="artifacts/distilbert_spam_model")
    parser.add_argument("--metrics_output", default="artifacts/metrics/distilbert_metrics.json")
    parser.add_argument("--experiment_name", default="Spam_Classification_DistilBERT")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    train_distilbert(
        train_path=args.train_path,
        test_path=args.test_path,
        model_output_dir=args.model_output_dir,
        metrics_output=args.metrics_output,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
    )