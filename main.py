from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data.collectors.phishtank_collector import PhishingDataCollector
from src.evaluation import ModelEvaluator
from src.features import DataPreprocessor, URLFeatureExtractor
from src.models import PhishingRepresentationModel, TabularMLPBaseline
from src.training import PhishingTrainer, TrainerConfig


def make_loader(X, y, batch_size: int = 64, shuffle: bool = True, input_dtype: str = "long") -> DataLoader:
    if input_dtype == "float":
        x_tensor = torch.FloatTensor(X)
    else:
        x_tensor = torch.LongTensor(X)
    dataset = TensorDataset(x_tensor, torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_url_dataset(args) -> pd.DataFrame:
    collector = PhishingDataCollector()

    print("[INFO] Collecting phishing URLs from PhishTank...")
    phish_df = collector.collect_phishing_urls(limit=args.max_phishing)

    print(f"[INFO] Loading legitimate domains from {args.legit_csv}...")
    legit_df = collector.collect_legitimate_urls(args.legit_csv, limit=args.max_legit)

    combined = pd.concat([phish_df[["url", "label"]], legit_df[["url", "label"]]], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return combined


def load_tabular_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Tabular dataset not found: {path}")
    return pd.read_csv(path)


def evaluate_and_save_artifacts(model, test_loader, device: str, output_dir: str):
    model.eval()
    y_true, y_pred, y_prob, reprs = [], [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits, _, _, representation = model(x_batch.to(device), return_repr=True)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())
            reprs.extend(representation.cpu().numpy().tolist())

    evaluator = ModelEvaluator(output_dir=output_dir)
    cm_path = evaluator.plot_confusion_matrix(y_true, y_pred)
    roc_path = evaluator.plot_roc_curve(y_true, y_prob)
    tsne_path = evaluator.visualize_representations(reprs, y_true)

    cm = confusion_matrix(y_true, y_pred)
    print("[INFO] Confusion matrix:\n", cm)
    print(f"[INFO] Saved: {cm_path}")
    if roc_path is not None:
        print(f"[INFO] Saved: {roc_path}")
    else:
        print("[INFO] Skipped ROC curve (test set has only one class).")

    if tsne_path is not None:
        print(f"[INFO] Saved: {tsne_path}")
    else:
        print("[INFO] Skipped t-SNE visualization (not enough samples).")


def parse_args():
    parser = argparse.ArgumentParser(description="Phishing detection with representation learning")
    parser.add_argument("--dataset-mode", type=str, default="url", choices=["url", "tabular"])
    parser.add_argument("--legit-csv", type=str, default="data/raw/top-1m.csv", help="Path to CSV of legit domains")
    parser.add_argument(
        "--tabular-csv",
        type=str,
        default="data/raw/Phishing_Legitimate_full.csv",
        help="Path to tabular phishing dataset CSV",
    )
    parser.add_argument("--label-col", type=str, default="CLASS_LABEL", help="Label column for tabular dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--max-phishing", type=int, default=50000)
    parser.add_argument("--max-legit", type=int, default=50000)
    parser.add_argument("--checkpoint", type=str, default="experiments/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--log-interval", type=int, default=100, help="Print training progress every N batches")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Dataset mode: {args.dataset_mode}")

    preprocessor = DataPreprocessor(max_len=args.max_len)
    input_dtype = "long"

    if args.dataset_mode == "tabular":
        print(f"[INFO] Loading tabular dataset from {args.tabular_csv}...")
        tabular_df = load_tabular_dataset(args.tabular_csv)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.prepare_tabular_dataset(
            tabular_df,
            label_col=args.label_col,
        )
        input_dtype = "float"
        model = TabularMLPBaseline(input_dim=X_train.shape[1])
        if args.checkpoint == "experiments/best_model.pt":
            args.checkpoint = "experiments/best_model_tabular.pt"
    else:
        df = load_url_dataset(args)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.prepare_dataset(df)
        model = PhishingRepresentationModel(vocab_size=URLFeatureExtractor.vocab_size())

    train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True, input_dtype=input_dtype)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False, input_dtype=input_dtype)
    test_loader = make_loader(X_test, y_test, batch_size=args.batch_size, shuffle=False, input_dtype=input_dtype)

    print(
        f"[INFO] Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)} | "
        f"batches(train)={len(train_loader)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    config = TrainerConfig(epochs=args.epochs, log_interval=args.log_interval)
    trainer = PhishingTrainer(model=model, device=device, config=config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_result = trainer.train(train_loader, val_loader, checkpoint_path=args.checkpoint)
    print(f"[INFO] Best validation F1: {train_result['best_val_f1']:.4f}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_metrics = trainer.evaluate(test_loader)

    print("\n===== TEST RESULTS =====")
    for key, value in test_metrics.items():
        print(f"{key:10s}: {value:.4f}")

    evaluate_and_save_artifacts(model, test_loader, device=device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
