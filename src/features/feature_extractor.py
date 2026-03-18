from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import tldextract


class URLFeatureExtractor:
    """Extract lexical features and char-level token sequences from URLs."""

    _chars = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
    _char2idx = {c: i + 1 for i, c in enumerate(_chars)}

    def extract_lexical_features(self, url: str) -> dict[str, float]:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        domain = ext.domain or ""

        return {
            "url_length": float(len(url)),
            "domain_length": float(len(domain)),
            "path_length": float(len(parsed.path or "")),
            "num_dots": float(url.count(".")),
            "num_hyphens": float(url.count("-")),
            "num_underscores": float(url.count("_")),
            "num_slashes": float(url.count("/")),
            "num_at": float(url.count("@")),
            "num_digits": float(sum(ch.isdigit() for ch in url)),
            "domain_entropy": float(self._entropy(domain)),
            "has_ip": float(bool(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", parsed.netloc or ""))),
            "has_https": float(url.startswith("https")),
            "has_www": float("www." in url),
            "tld_suspicious": float(ext.suffix in {"tk", "ml", "ga", "cf", "gq"}),
        }

    def url_to_char_sequence(self, url: str, max_len: int = 200) -> list[int]:
        normalized = (url or "").lower()[:max_len]
        sequence = [self._char2idx.get(char, 0) for char in normalized]
        padding = [0] * max(0, max_len - len(sequence))
        return sequence + padding

    @classmethod
    def vocab_size(cls) -> int:
        return len(cls._chars) + 1

    @staticmethod
    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        probabilities = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)
