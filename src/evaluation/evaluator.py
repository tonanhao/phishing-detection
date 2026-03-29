from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, roc_curve


class ModelEvaluator:
    """Visualization helpers for classification results and learned representations."""

    def __init__(self, output_dir: str = "experiments") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, filename: str = "confusion_matrix.png") -> Path:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_roc_curve(self, y_true, y_score, filename: str = "roc_curve.png") -> Path | None:
        if len(set(y_true)) < 2:
            return None

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Phishing Detection")
        ax.legend(loc="lower right")

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def visualize_representations(self, representations, labels, filename: str = "tsne_representations.png") -> Path | None:
        repr_array = np.asarray(representations, dtype=np.float32)
        n_samples = repr_array.shape[0] if repr_array.ndim == 2 else 0
        if n_samples < 3:
            return None

        perplexity = float(min(30, n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
        repr_2d = tsne.fit_transform(repr_array)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(repr_2d[:, 0], repr_2d[:, 1], c=labels, cmap="RdYlGn", alpha=0.6)
        fig.colorbar(scatter, label="0=Phishing, 1=Legitimate")
        ax.set_title("Learned URL Representations (t-SNE)")

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
