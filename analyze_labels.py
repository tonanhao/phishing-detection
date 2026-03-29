import pandas as pd

df = pd.read_csv('data/raw/PhiUSIIL_Phishing_URL_Dataset.csv', nrows=5000)

print("=== Label Distribution ===")
print(df['label'].value_counts())

print("\n=== Label 0 (Phishing?) - Sample URLs ===")
for url in df[df['label']==0]['URL'].head(5):
    print(f"  {url}")

print("\n=== Label 1 (Legitimate?) - Sample URLs ===")  
for url in df[df['label']==1]['URL'].head(5):
    print(f"  {url}")

print("\n=== Suspicious TLD distribution by label ===")
suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top']
for tld in suspicious_tlds:
    count_0 = df[(df['URL'].str.endswith('.' + tld)) & (df['label']==0)].shape[0]
    count_1 = df[(df['URL'].str.endswith('.' + tld)) & (df['label']==1)].shape[0]
    if count_0 + count_1 > 0:
        print(f"  .{tld}: label=0: {count_0}, label=1: {count_1}")
