from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import ModelEvaluator
from src.features.load_phiusiiil import prepare_for_training
from src.models import TabularMLPBaseline
from src.training import PhishingTrainer, TrainerConfig


def make_loader(X, y, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    x_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(x_tensor, torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    parser = argparse.ArgumentParser(description="Phishing detection training with PhiUSIIL dataset")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
        help="Path to PhiUSIIL dataset CSV",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="experiments/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading PhiUSIIL dataset from {args.csv}...")
    data = prepare_for_training(args.csv)

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    print(
        f"[INFO] Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)} | "
        f"Features: {X_train.shape[1]}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = TabularMLPBaseline(input_dim=X_train.shape[1])

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
