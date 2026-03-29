"""
Utility to load and preprocess PhishUSIIL Phishing URL Dataset.

Dataset: PhiUSIIL_Phishing_URL_Dataset.csv
Source: https://www.kaggle.com/datasets/abdmental ridd/phiusiiil-phishing-url-dataset

Columns:
    - URL: The actual URL string
    - label: 0 (legitimate) or 1 (phishing)
    - 54 other features extracted from URL and web page
"""

import pandas as pd
from pathlib import Path


def load_phiusiiil_dataset(
    csv_path: str = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
    include_features: bool = True,
) -> pd.DataFrame:
    """
    Load PhiUSIIL dataset and return in standardized format.

    Args:
        csv_path: Path to the CSV file
        include_features: If True, keep all feature columns.
                         If False, only keep URL and label.

    Returns:
        DataFrame with columns: ['url', 'label', *features]
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print(f"[INFO] Loaded PhiUSIIL dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"[INFO] Label distribution:")
    print(df['label'].value_counts())

    # Check for duplicate URLs
    duplicates = df.duplicated(subset=['URL'], keep='first').sum()
    if duplicates > 0:
        print(f"[WARNING] Found {duplicates} duplicate URLs, removing duplicates...")
        df = df.drop_duplicates(subset=['URL'], keep='first')
        print(f"[INFO] After removing duplicates: {len(df)} rows")

    # Standardize column names
    df = df.rename(columns={'URL': 'url'})

    # Keep only needed columns
    if not include_features:
        df = df[['url', 'label']]

    return df


# Features that leak label - derived from page content analysis or too predictive
# These give away the answer, not from URL itself
LEAKING_FEATURES = {
    # Target categories - perfect predictors
    'Bank', 'Pay', 'Crypto',
    # Form/HTML features - page content
    'HasPasswordField', 'HasExternalFormSubmit', 'HasSubmitButton',
    'HasHiddenFields', 'InsecureForms', 'PopUpWindow', 'SubmitInfoToEmail',
    'NoOfPopup', 'NoOfiFrame', 'IframeOrFrame',
    # Domain/Page analysis features
    'HasTitle', 'Title', 'DomainTitleMatchScore', 'URLTitleMatchScore',
    'HasFavicon', 'Robots', 'IsResponsive',
    # Content features - require crawling
    'HasDescription', 'HasCopyrightInfo',
    'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef',
    'PctExtHyperlinks', 'PctExtResourceUrls',
    'LineOfCode', 'LargestLineLength',
    # Features that need page rendering
    'FakeLinkInStatusBar', 'RightClickDisabled',
    'ImagesOnlyInForm', 'MissingTitle',
    # Other suspicious features
    'EmbeddedBrandName', 'FrequentDomainNameMismatch',
    'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
    'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT',
    'PctNullSelfRedirectHyperlinks', 'PctNullSelfRedirectHyperlinksRT',
    'PctExtResourceUrlsRT',
}


# Only keep URL-derived features (no page crawling needed)
URL_ONLY_FEATURES = {
    'URLLength', 'DomainLength', 'TLDLength',
    'NumDots', 'NumDash', 'NumDashInHostname',
    'AtSymbol', 'TildeSymbol', 'NumUnderscore',
    'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
    'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
    'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
    'HostnameLength', 'PathLength', 'QueryLength',
    'DoubleSlashInPath', 'NoOfSubDomain',
    'CharContinuationRate', 'URLCharProb', 'TLDLegitimateProb',
    'URLSimilarityIndex', 'IsDomainIP', 'HasObfuscation',
    'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL',
    'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
    'IsHTTPS',
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Get only URL-derived features (no page content features).

    Args:
        df: DataFrame loaded from PhiUSIIL dataset

    Returns:
        List of feature column names - only from URL string
    """
    feature_cols = []
    for col in df.columns:
        if col in URL_ONLY_FEATURES and df[col].dtype in ['int64', 'float64']:
            feature_cols.append(col)

    print(f"[INFO] Using {len(feature_cols)} URL-only features (no page crawling features)")
    return feature_cols

    print(f"[INFO] Excluded {len(LEAKING_FEATURES)} leaking features: {sorted(LEAKING_FEATURES)}")
    return feature_cols


def prepare_for_training(
    csv_path: str = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Load dataset and prepare train/val/test splits.

    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = load_phiusiiil_dataset(csv_path, include_features=True)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df['label'].values

    # Split: 70% train, 20% test, 10% val
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y if len(set(y)) > 1 else None,
    )

    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - relative_val,
        random_state=random_state,
        stratify=y_temp if len(set(y_temp)) > 1 else None,
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[INFO] Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"[INFO] Features: {len(feature_cols)}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler,
    }


if __name__ == "__main__":
    # Quick test
    df = load_phiusiil_dataset()
    print(f"\nColumns: {df.columns.tolist()[:10]}...")
    print(f"\nSample URL: {df['url'].iloc[0]}")
    print(f"Sample label: {df['label'].iloc[0]}")
