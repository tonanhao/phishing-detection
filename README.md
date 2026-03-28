# Phishing Detection - Hướng dẫn sử dụng

Dự án này huấn luyện mô hình phát hiện URL phishing với 2 chế độ:

- `url`: học biểu diễn trực tiếp từ chuỗi URL (representation learning).
- `tabular`: MLP baseline cho dữ liệu đã trích xuất đặc trưng.

## 1. Yêu cầu

- Python 3.10+
- `pip` phiên bản mới
- Khuyến nghị có GPU CUDA nếu huấn luyện dữ liệu lớn

## 2. Cấu trúc dự án

```text
phishing-detection/
|-- main.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   |   |-- top-1m.csv
|   |   |-- Phishing_Legitimate_full.csv
|   `-- collectors/
|       `-- phishtank_collector.py
|-- src/
|   |-- features/
|   |-- models/
|   |-- training/
|   `-- evaluation/
`-- experiments/
```

## 3. Cài đặt môi trường

### Windows (Git Bash)

```bash
cd /c/IT/ATHTMMT/Project/phishing-detection
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd c:\IT\ATHTMMT\Project\phishing-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ghi chú:
- Nếu cài `torch-geometric` lỗi do CUDA/OS, bạn có thể bỏ qua tạm thời nếu chưa dùng module liên quan.
- Nếu muốn dùng GPU, cài PyTorch theo hướng dẫn chính thức: https://pytorch.org/get-started/locally/

## 4. Chuẩn bị dữ liệu

### 4.1 Dữ liệu hợp lệ (bắt buộc cho `url` mode)

Tạo file `data/raw/top-1m.csv` với định dạng:

```text
1,google.com
2,youtube.com
3,facebook.com
```

### 4.2 Dữ liệu phishing

Trong `url` mode, hệ thống tự động tải feed phishing theo thứ tự:

1. PhishTank JSON
2. OpenPhish feed (fallback nếu PhishTank lỗi)

### 4.3 Dữ liệu tabular

Mặc định dùng file `data/raw/Phishing_Legitimate_full.csv`.

File cần có cột nhãn mặc định là `CLASS_LABEL` (có thể đổi bằng `--label-col`).

## 5. Chạy huấn luyện và đánh giá

File chạy chính: `main.py`

### 5.1 URL mode (mặc định)

```bash
python main.py \
  --dataset-mode url \
  --legit-csv data/raw/top-1m.csv \
  --epochs 20 \
  --batch-size 64 \
  --max-phishing 50000 \
  --max-legit 50000
```

### 5.2 Tabular mode

```bash
python main.py \
  --dataset-mode tabular \
  --tabular-csv data/raw/Phishing_Legitimate_full.csv \
  --label-col CLASS_LABEL \
  --epochs 20 \
  --batch-size 128
```

## 6. Tham số quan trọng

- `--dataset-mode`: `url` hoặc `tabular`
- `--legit-csv`: đường dẫn file domain hợp lệ (cho `url` mode)
- `--tabular-csv`: đường dẫn file CSV tabular
- `--label-col`: tên cột nhãn trong tabular CSV
- `--max-len`: độ dài URL sau padding/cắt (mặc định `200`)
- `--epochs`: số epoch train (mặc định `20`)
- `--batch-size`: kích thước mini-batch
- `--max-phishing`: giới hạn số mẫu phishing tải về
- `--max-legit`: giới hạn số mẫu hợp lệ
- `--checkpoint`: nơi lưu model tốt nhất
- `--output-dir`: thư mục lưu biểu đồ/kết quả
- `--log-interval`: in log train mỗi N batch

## 7. Kết quả sinh ra

Sau khi chạy xong, các file thường nằm trong `experiments/`:

- `best_model.pt` (cho URL mode)
- `best_model_tabular.pt` (cho tabular mode)
- `confusion_matrix.png`
- `roc_curve.png` (nếu test set có đủ 2 lớp)
- `tsne_representations.png` (nếu đủ mẫu để vẽ)

Console sẽ in các metric:
- `accuracy`
- `f1`
- `precision`
- `recall`

## 8. Ví dụ chạy nhanh

```bash
python main.py --dataset-mode tabular --epochs 3 --batch-size 64
```

Lệnh trên phù hợp để test pipeline nhanh trước khi train đầy đủ.

## 9. Lỗi thường gặp

- Báo lỗi không tìm thấy `top-1m.csv`:
  - Kiểm tra lại đường dẫn `--legit-csv`.
- Không tải được feed phishing:
  - Kiểm tra internet, proxy, firewall.
  - Thử chạy lại sau và giảm `--max-phishing` để test.
- Out-of-memory trên GPU:
  - Giảm `--batch-size`.
  - Giảm `--max-len` (cho URL mode).

## 10. Tài liệu liên quan

- Hướng dẫn chi tiết: `HUONG_DAN_CAI_DAT_CAU_HINH.md`
