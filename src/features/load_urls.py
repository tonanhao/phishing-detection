"""
Load URLs for training - URL mode (no feature extraction).
"""

import pandas as pd
from pathlib import Path


def load_legitimate_urls(csv_path: str, limit: int = 50000) -> pd.DataFrame:
    """
    Load legitimate URLs from Alexa top-1M or similar CSV.
    
    Expected format: rank,domain (no header or with header)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Legitimate URLs file not found: {path}")
    
    # Try different formats
    try:
        df = pd.read_csv(path, names=['rank', 'domain'], header=None)
    except:
        df = pd.read_csv(path)
        # If has header, rename
        if 'domain' not in df.columns:
            df.columns = ['rank', 'domain']
    
    # Convert to URL format
    df['url'] = 'http://' + df['domain'].astype(str)
    df['label'] = 0  # 0 = legitimate
    
    print(f"[INFO] Loaded {len(df)} legitimate URLs")
    return df[['url', 'label']].head(limit).reset_index(drop=True)


def load_phishing_urls_from_phiussiil(csv_path: str, limit: int = 50000) -> pd.DataFrame:
    """
    Load phishing URLs from PhiUSIIL dataset.
    Based on our analysis: label=0 means phishing in this dataset.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"PhiUSIIL dataset not found: {path}")
    
    df = pd.read_csv(path)
    
    # Filter to phishing only (label=0 in PhiUSIIL = phishing)
    phishing_df = df[df['label'] == 0][['URL', 'label']].copy()
    phishing_df.columns = ['url', 'label']
    
    print(f"[INFO] Loaded {len(phishing_df)} phishing URLs from PhiUSIIL")
    return phishing_df.head(limit).reset_index(drop=True)


def load_phishing_urls_from_csv(csv_path: str, url_column: str = 'url', label_column: str = 'label', 
                                  phishing_value: int = 1, limit: int = 50000) -> pd.DataFrame:
    """
    Load phishing URLs from custom CSV file.
    
    Args:
        csv_path: Path to CSV file
        url_column: Name of URL column
        label_column: Name of label column
        phishing_value: Value that indicates phishing (default=1)
        limit: Maximum number of URLs to load
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Phishing URLs file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Find URL and label columns
    if url_column not in df.columns:
        # Try common names
        for col in ['URL', 'url', 'Domain', 'domain', 'link', 'Link']:
            if col in df.columns:
                url_column = col
                break
    
    if label_column not in df.columns:
        label_column = 'label'
    
    # Filter phishing URLs
    phishing_df = df[df[label_column] == phishing_value][[url_column, label_column]].copy()
    phishing_df.columns = ['url', 'label']
    
    print(f"[INFO] Loaded {len(phishing_df)} phishing URLs from {csv_path}")
    return phishing_df.head(limit).reset_index(drop=True)


def load_all_urls(
    legit_csv: str = "data/raw/top-1m.csv",
    phishing_csv: str = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
    max_legit: int = 50000,
    max_phishing: int = 50000,
    phishing_source: str = "phiussiil",  # "phiussiil" or "csv"
) -> pd.DataFrame:
    """
    Load both legitimate and phishing URLs and combine.
    
    Args:
        legit_csv: Path to legitimate URLs CSV (top-1M)
        phishing_csv: Path to phishing URLs CSV
        max_legit: Max legitimate URLs
        max_phishing: Max phishing URLs
        phishing_source: "phiussiil" or "csv"
    
    Returns:
        Combined DataFrame with columns: url, label
    """
    # Load legitimate
    legit_df = load_legitimate_urls(legit_csv, limit=max_legit)
    
    # Load phishing
    if phishing_source == "phiussiil":
        phishing_df = load_phishing_urls_from_phiussiil(phishing_csv, limit=max_phishing)
    else:
        phishing_df = load_phishing_urls_from_csv(phishing_csv, limit=max_phishing)
    
    # Combine and shuffle
    combined = pd.concat([legit_df, phishing_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"[INFO] Combined dataset: {len(combined)} URLs")
    print(f"  - Legitimate (label=0): {(combined['label']==0).sum()}")
    print(f"  - Phishing (label=1): {(combined['label']==1).sum()}")
    
    return combined
