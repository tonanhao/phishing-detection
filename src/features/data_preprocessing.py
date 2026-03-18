from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.feature_extractor import URLFeatureExtractor


@dataclass
class DataPreprocessor:
    """Prepare URL sequence tensors for deep learning models."""

    max_len: int = 200
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        self.extractor = URLFeatureExtractor()
        self.tabular_scaler: StandardScaler | None = None

    def prepare_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        required = {"url", "label"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        cleaned = df[["url", "label"]].dropna().copy()
        cleaned["url"] = cleaned["url"].astype(str)
        cleaned["label"] = cleaned["label"].astype(int)

        X = np.array(
            [self.extractor.url_to_char_sequence(url, self.max_len) for url in cleaned["url"].tolist()],
            dtype=np.int64,
        )
        y = cleaned["label"].to_numpy(dtype=np.int64)

        initial_stratify = y if self._can_stratify(y) else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=initial_stratify,
        )

        relative_val = self.val_size / (self.test_size + self.val_size)
        second_stratify = y_temp if self._can_stratify(y_temp) else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - relative_val,
            random_state=self.random_state,
            stratify=second_stratify,
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def prepare_tabular_dataset(
        self,
        df: pd.DataFrame,
        label_col: str = "CLASS_LABEL",
        drop_columns: tuple[str, ...] = ("id",),
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")

        cleaned = df.copy()
        cleaned = cleaned.dropna(subset=[label_col])
        cleaned[label_col] = pd.to_numeric(cleaned[label_col], errors="coerce")
        cleaned = cleaned.dropna(subset=[label_col])
        cleaned[label_col] = cleaned[label_col].astype(int)

        feature_df = cleaned.drop(columns=[label_col], errors="ignore")
        for col in drop_columns:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=[col])

        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
        feature_df = feature_df.dropna(axis=0, how="any")

        y = cleaned.loc[feature_df.index, label_col].to_numpy(dtype=np.int64)
        X = feature_df.to_numpy(dtype=np.float32)

        initial_stratify = y if self._can_stratify(y) else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=initial_stratify,
        )

        relative_val = self.val_size / (self.test_size + self.val_size)
        second_stratify = y_temp if self._can_stratify(y_temp) else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - relative_val,
            random_state=self.random_state,
            stratify=second_stratify,
        )

        self.tabular_scaler = StandardScaler()
        X_train = self.tabular_scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.tabular_scaler.transform(X_val).astype(np.float32)
        X_test = self.tabular_scaler.transform(X_test).astype(np.float32)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @staticmethod
    def _can_stratify(labels: np.ndarray) -> bool:
        counts = np.bincount(labels)
        non_zero_counts = counts[counts > 0]
        return bool(len(non_zero_counts) > 1 and non_zero_counts.min() >= 2)
