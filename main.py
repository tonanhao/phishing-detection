from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import ModelEvaluator
from src.features import DataPreprocessor, URLFeatureExtractor
from src.features.load_phiusiiil import prepare_for_training
from src.features.load_urls import load_all_urls
from src.models import PhishingRepresentationModel, TabularMLPBaseline
from src.training import PhishingTrainer, TrainerConfig


def make_loader(X, y, batch_size: int = 64, shuffle: bool = True, input_dtype: str = "float") -> DataLoader:
    if input_dtype == "float":
        x_tensor = torch.FloatTensor(X)
    else:
        x_tensor = torch.LongTensor(X)
    dataset = TensorDataset(x_tensor, torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_url_data(args) -> tuple:
    """
    Load URLs and prepare for training in URL mode.
    """
    df = load_all_urls(
        legit_csv=args.legit_csv,
        phishing_csv=args.phishing_csv,
        max_legit=args.max_legit,
        max_phishing=args.max_phishing,
        phishing_source=args.phishing_source,
    )
    
    # Tokenize URLs
    extractor = URLFeatureExtractor()
    X = np.array([extractor.url_to_char_sequence(url, args.max_len) for url in df['url']], dtype=np.int64)
    y = df['label'].values.astype(np.int64)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=42, 
        stratify=y_temp if len(set(y_temp)) > 1 else None
    )
    
    print(f"[INFO] Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


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
    parser = argparse.ArgumentParser(description="Phishing detection training")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="tabular", choices=["tabular", "url"],
                       help="Training mode: tabular (MLP with features) or url (CNN+BiLSTM with raw URLs)")
    
    # Tabular mode
    parser.add_argument("--csv", type=str, default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
                       help="Path to dataset CSV")
    
    # URL mode
    parser.add_argument("--legit-csv", type=str, default="data/raw/top-1m.csv",
                       help="Path to legitimate URLs (top-1M)")
    parser.add_argument("--phishing-csv", type=str, default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
                       help="Path to phishing URLs CSV")
    parser.add_argument("--phishing-source", type=str, default="phiussiil",
                       choices=["phiussiil", "csv"],
                       help="Source of phishing URLs: phiussiil or csv")
    parser.add_argument("--max-legit", type=int, default=50000,
                       help="Max legitimate URLs")
    parser.add_argument("--max-phishing", type=int, default=50000,
                       help="Max phishing URLs")
    parser.add_argument("--max-len", type=int, default=200,
                       help="Max URL length for tokenization")
    
    # Training options
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="experiments/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--log-interval", type=int, default=100)
    
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Mode: {args.mode}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    if args.mode == "tabular":
        # Tabular mode: MLP with pre-extracted features
        print(f"[INFO] Loading tabular dataset from {args.csv}...")
        data = prepare_for_training(args.csv)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        
        input_dtype = "float"
        model = TabularMLPBaseline(input_dim=X_train.shape[1])
        checkpoint = args.checkpoint.replace(".pt", "_tabular.pt")
        
    else:
        # URL mode: CNN+BiLSTM with raw URLs
        print(f"[INFO] Loading URLs for URL mode...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_url_data(args)
        
        input_dtype = "long"
        model = PhishingRepresentationModel(vocab_size=URLFeatureExtractor.vocab_size())
        checkpoint = args.checkpoint.replace(".pt", "_url.pt")
    
    # Create dataloaders
    train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True, input_dtype=input_dtype)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False, input_dtype=input_dtype)
    test_loader = make_loader(X_test, y_test, batch_size=args.batch_size, shuffle=False, input_dtype=input_dtype)
    
    if args.mode == "tabular":
        print(f"[INFO] Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} | Features: {X_train.shape[1]}")
    else:
        print(f"[INFO] Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} | URL length: {args.max_len}")
    
    # Train
    config = TrainerConfig(epochs=args.epochs, log_interval=args.log_interval)
    trainer = PhishingTrainer(model=model, device=device, config=config)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_result = trainer.train(train_loader, val_loader, checkpoint_path=checkpoint)
    print(f"[INFO] Best validation F1: {train_result['best_val_f1']:.4f}")
    
    # Evaluate
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n===== TEST RESULTS =====")
    for key, value in test_metrics.items():
        print(f"{key:10s}: {value:.4f}")
    
    evaluate_and_save_artifacts(model, test_loader, device=device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
