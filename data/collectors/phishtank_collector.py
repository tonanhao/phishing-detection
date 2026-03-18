from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


@dataclass
class PhishingDataCollector:
    """Collect phishing and legitimate URLs from common public sources."""

    phishtank_urls: tuple[str, ...] = (
        "https://data.phishtank.com/data/online-valid.json",
        "https://data.phishtank.com/data/online-valid.csv",
    )
    openphish_url: str = "https://openphish.com/feed.txt"

    def _collect_from_phishtank_json(self) -> pd.DataFrame:
        last_error: Exception | None = None
        for url in self.phishtank_urls:
            # Skip CSV endpoint in this method.
            if not url.lower().endswith(".json"):
                continue
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                rows = []
                for entry in data:
                    rows.append(
                        {
                            "url": entry.get("url", ""),
                            "label": 1,
                            "target": entry.get("target", "unknown"),
                            "verified": bool(entry.get("verified", False)),
                        }
                    )

                phishing_df = pd.DataFrame(rows)
                if not phishing_df.empty:
                    phishing_df = phishing_df[phishing_df["url"].astype(str).str.len() > 0]
                    return phishing_df.reset_index(drop=True)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError(f"PhishTank JSON feed failed: {last_error}") from last_error
        raise RuntimeError("PhishTank JSON feed failed: no usable endpoint")

    def _collect_from_openphish(self) -> pd.DataFrame:
        response = requests.get(self.openphish_url, timeout=30)
        response.raise_for_status()

        urls = [line.strip() for line in response.text.splitlines() if line.strip()]
        return pd.DataFrame(
            {
                "url": urls,
                "label": 1,
                "target": "unknown",
                "verified": True,
            }
        )

    def collect_phishing_urls(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Collect phishing URLs with fallback sources if one feed is down."""
        errors: list[str] = []

        try:
            phishing_df = self._collect_from_phishtank_json()
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            try:
                phishing_df = self._collect_from_openphish()
            except Exception as openphish_exc:  # noqa: BLE001
                errors.append(f"OpenPhish feed failed: {openphish_exc}")
                joined = " | ".join(errors)
                raise RuntimeError(
                    "Unable to collect phishing URLs from public feeds. "
                    f"Details: {joined}"
                ) from openphish_exc

        phishing_df = phishing_df[phishing_df["url"].astype(str).str.len() > 0]

        if limit is not None:
            phishing_df = phishing_df.head(limit)
        return phishing_df.reset_index(drop=True)

    def collect_legitimate_urls(self, csv_path: str | Path, limit: int = 50000) -> pd.DataFrame:
        """Load top domains and map them into legitimate URLs."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Legitimate domain file not found: {csv_path}")

        legit_df = pd.read_csv(csv_path, names=["rank", "domain"])
        legit_df["url"] = "http://" + legit_df["domain"].astype(str)
        legit_df["label"] = 0
        return legit_df[["url", "label"]].head(limit).reset_index(drop=True)


if __name__ == "__main__":
    collector = PhishingDataCollector()
    sample = collector.collect_phishing_urls(limit=5)
    print(sample.head())
