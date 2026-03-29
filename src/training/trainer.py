from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    reconstruction_weight: float = 0.1
    max_grad_norm: float = 1.0
    epochs: int = 20
    log_interval: int = 100


class PhishingTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu", config: TrainerConfig | None = None) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config or TrainerConfig()

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(self.config.epochs, 1))

    def compute_loss(self, logits: torch.Tensor, reconstruction: torch.Tensor, fused: torch.Tensor, labels: torch.Tensor):
        cls_loss = self.ce_loss(logits, labels)
        rec_loss = self.mse_loss(reconstruction, fused.detach())
        total = cls_loss + self.config.reconstruction_weight * rec_loss
        return total, cls_loss.detach().item(), rec_loss.detach().item()

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (batch_x, batch_y) in enumerate(dataloader, start=1):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits, reconstruction, fused = self.model(batch_x)
            loss, _, _ = self.compute_loss(logits, reconstruction, fused, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            running_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

            if self.config.log_interval > 0 and step % self.config.log_interval == 0:
                avg_so_far = running_loss / step
                acc_so_far = correct / max(total, 1)
                print(
                    f"  [train] step {step}/{len(dataloader)} | "
                    f"loss={avg_so_far:.4f} | acc={acc_so_far:.4f}"
                )

        self.scheduler.step()
        avg_loss = running_loss / max(len(dataloader), 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict[str, float]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch_x, batch_y in dataloader:
            logits, _, _ = self.model(batch_x.to(self.device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())

        if not all_labels:
            return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        return {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "f1": float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
            "precision": float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        }

    def train(self, train_loader, val_loader, checkpoint_path: str | Path = "best_model.pt") -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        best_f1 = -np.inf
        history = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, **val_metrics})

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | val_f1={val_metrics['f1']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

        return {"best_val_f1": float(best_f1), "history": history}
