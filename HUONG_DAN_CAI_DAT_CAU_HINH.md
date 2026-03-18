# Huong Dan Cai Dat Va Cau Hinh

Tai lieu nay huong dan cai dat moi truong va chay he thong phishing detection bang representation learning.

## 1. Yeu cau he thong

- Python 3.10 hoac moi hon
- pip moi nhat
- (Khuyen nghi) GPU CUDA neu train du lieu lon

## 2. Tao moi truong ao

### Tren Windows (PowerShell)

```powershell
cd c:\IT\ATHTMMT\Project\phishing-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Tren Windows (Git Bash)

```bash
cd /c/IT/ATHTMMT/Project/phishing-detection
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

## 3. Cai dat thu vien

```bash
pip install -r requirements.txt
```

Ghi chu:
- Neu `torch-geometric` khong cai duoc do phien ban CUDA, co the tam thoi bo qua neu ban chua dung GNN.
- Neu dung GPU, cai dat ban PyTorch phu hop tai: https://pytorch.org/get-started/locally/

## 4. Chuan bi du lieu

1. Tao file domain hop le tai `data/raw/top-1m.csv`.
2. Dinh dang file `top-1m.csv`:

```text
1,google.com
2,youtube.com
3,facebook.com
```

3. URL phishing duoc lay tu PhishTank feed trong qua trinh chay.

## 5. Chay huan luyen va danh gia

He thong ho tro 2 che do du lieu:

- `url`: Representation learning tu URL raw (che do cu)
- `tabular`: MLP baseline cho dataset da trich xuat dac trung (49 features + CLASS_LABEL)

### 5.1 Che do URL (mac dinh)

```bash
python main.py \
  --dataset-mode url \
  --legit-csv data/raw/top-1m.csv \
  --epochs 20 \
  --batch-size 64 \
  --max-phishing 50000 \
  --max-legit 50000
```

### 5.2 Che do Tabular

```bash
python main.py \
  --dataset-mode tabular \
  --tabular-csv data/raw/Phishing_Legitimate_full.csv \
  --label-col CLASS_LABEL \
  --epochs 20 \
  --batch-size 128
```

## 6. Cau hinh tham so quan trong

- `--max-len`: Do dai URL sau padding/cat (mac dinh 200)
- `--dataset-mode`: `url` hoac `tabular`
- `--tabular-csv`: Duong dan file CSV tabular
- `--label-col`: Ten cot nhan trong che do tabular
- `--epochs`: So epoch train (mac dinh 20)
- `--batch-size`: Kich thuoc mini-batch
- `--checkpoint`: Noi luu trong so model tot nhat
- `--output-dir`: Thu muc luu bieu do va ket qua

## 7. Dau ra sau khi chay

- Model tot nhat: `experiments/best_model.pt`
- Confusion matrix: `experiments/confusion_matrix.png`
- ROC curve: `experiments/roc_curve.png`
- t-SNE representation: `experiments/tsne_representations.png`

## 8. Xu ly loi thuong gap

- Loi khong tim thay `top-1m.csv`:
  - Kiem tra lai duong dan `--legit-csv`.
- Loi khong tai duoc PhishTank:
  - Kiem tra internet/firewall.
  - Chay lai sau, hoac thay nguon du lieu phishing offline.
- Loi thieu bo nho GPU:
  - Giam `--batch-size` hoac giam `--max-len`.
