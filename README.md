# Deep Learning – Final Project (NEU, 2026)

Ba task, task nào cũng đánh giá bằng R², NSE, MAE, RMSE:

| Task | Script | Dataset | Mô hình |
|---|---|---|---|
| 1 | `02_task1/task1_rnn_attention.py` | ETTh1, dự báo `OT` tại t+2 | LSTM + Additive (Bahdanau) attention |
| 2 | `03_task2/task2_seq2seq_attention.py` | ETTh1, dự báo `OT` tại t+1…t+5 | Encoder–Decoder LSTM + Luong attention |
| 3 | `04_task3/task3_image_to_sequence.py` | SVHN (số nhà nhiều chữ số) | CNN encoder + LSTM decoder + attention |

Báo cáo: `05_report/output/Final_Report.docx`, được sinh tự động bởi `05_report/build_report.py`. Số liệu lấy từ các file `output/results.json`.

## 1. Cài môi trường

Đã chạy với Python 3.14, Keras 3 trên backend PyTorch.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Không có GPU thì bỏ `--extra-index-url` và đổi dòng `torch==...+cu128` thành `torch`. Task 1 và 2 chạy CPU vẫn ổn, còn Task 3 sẽ chậm.

## 2. Tải dữ liệu (không có trong repo)

| Dataset | Link | Đặt vào |
|---|---|---|
| ETTh1 | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv | `01_data/raw/etth1/ETTh1.csv` |
| SVHN format 1, train | http://ufldl.stanford.edu/housenumbers/train.tar.gz | `01_data/raw/svhn/train.tar.gz` |
| SVHN format 1, test | http://ufldl.stanford.edu/housenumbers/test.tar.gz | `01_data/raw/svhn/test.tar.gz` |

```bash
mkdir -p 01_data/raw/etth1 01_data/raw/svhn
curl -L -o 01_data/raw/etth1/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
curl -L -o 01_data/raw/svhn/train.tar.gz http://ufldl.stanford.edu/housenumbers/train.tar.gz
curl -L -o 01_data/raw/svhn/test.tar.gz http://ufldl.stanford.edu/housenumbers/test.tar.gz
```

Chỉ cần tải file gốc về rồi chạy script. Giải nén, cắt ảnh và chia train/val/test đều nằm trong code, không có bước nào phải làm tay.

## 3. Chạy

Luôn chạy từ thư mục gốc repo, và chạy từng script một (Task 3 tốn nhiều RAM).

```bash
# Git Bash
PYTHONUTF8=1 .venv/Scripts/python 01_data/prepare_svhn.py        # tạo 01_data/processed/svhn_64x128.npz
PYTHONUTF8=1 .venv/Scripts/python 02_task1/task1_rnn_attention.py
PYTHONUTF8=1 .venv/Scripts/python 03_task2/task2_seq2seq_attention.py
PYTHONUTF8=1 .venv/Scripts/python 04_task3/task3_image_to_sequence.py  # ~65 phút trên RTX 5060
PYTHONUTF8=1 .venv/Scripts/python 05_report/build_report.py
```

Trên PowerShell, chạy `$env:PYTHONUTF8=1` một lần trước rồi bỏ tiền tố `PYTHONUTF8=1`. Nếu thiếu biến này, các dòng `print` tiếng Việt sẽ lỗi `UnicodeEncodeError` khi output được ghi ra file.

## 4. Dữ liệu được xử lý thế nào (tất cả nằm trong code)

**ETTh1** (Task 1 và Task 2, phần đầu mỗi script):
- 17.420 dòng theo giờ, 7 cột: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT. Code thêm `hour_sin` và `hour_cos` (giờ trong ngày).
- Chia theo thời gian 12/4/4 tháng: train 8.640 dòng, val 2.880, test 2.880 (split chuẩn của benchmark ETT). 3.020 dòng cuối không dùng.
- `StandardScaler` chỉ fit trên tập train.
- Cửa sổ đầu vào là 24 giờ. Mô hình học **độ thay đổi** ΔOT = OT(t+k) − OT(t) chứ không học thẳng OT, vì giai đoạn test lạnh hơn train rất nhiều (OT trung bình 4,8 °C so với 17,1 °C). Kết quả được cộng lại OT(t) và inverse-scale trước khi tính metric.
- Baseline để so sánh là persistence: dự báo = OT(t).

**SVHN** (`01_data/prepare_svhn.py`, sau đó đến Task 3):
- Tự giải nén `train.tar.gz` và `test.tar.gz` nếu chưa có thư mục `train/` và `test/`.
- Đọc nhãn và bounding box trong `digitStruct.mat`. Nhãn 10 được hiểu là chữ số 0.
- Cắt vùng bao quanh tất cả chữ số, nới thêm 15% mỗi phía, rồi resize về 64×128 RGB.
- Bỏ các ảnh có hơn 5 chữ số (1 ảnh train). Còn lại 33.401 ảnh train và 13.068 ảnh test.
- Trong Task 3, 10% cuối của tập train được tách làm validation: train 30.061, val 3.340, test 13.068 (dùng nguyên tập test chính thức).

## 5. Kết quả (tập test)

| Task | Kết quả |
|---|---|
| 1 (t+2) | R² 0,938 · NSE 0,931 · MAE 0,604 · RMSE 0,822 °C (persistence: NSE 0,932, RMSE 0,814) |
| 2 (t+1…t+5, gộp) | R² 0,908 · NSE 0,903 · RMSE 0,972 °C (persistence: RMSE 0,999); tốt hơn persistence từ t+2 đến t+5 |
| 3 | Đúng cả chuỗi 90,3% · đúng từng chữ số 94,1% · trên giá trị số R² 0,714 (thang log: 0,912) |

Chi tiết xem trong `*/output/results.json`, log train trong `*/output.log`.

Lưu ý:
- R² ở đây là bình phương hệ số tương quan Pearson; NSE = 1 − SSE/SST. Hai chỉ số này khác nhau. `sklearn.metrics.r2_score` thực ra tính ra NSE.
- `04_task3/output/predictions.png` hiện có một ảnh bị lặp. Code đã sửa, hình sẽ đúng sau lần train Task 3 tiếp theo.

## Cấu trúc

```
docs/          đề bài, template báo cáo, code mẫu của thầy (chỉ để tham khảo)
01_data/       prepare_svhn.py; raw/ (tự tải), processed/ (sinh ra khi chạy)
02_task1/      Task 1 + output/
03_task2/      Task 2 + output/
04_task3/      Task 3 + output/
05_report/     build_report.py + output/Final_Report.docx
```

Code viết theo phong cách code mẫu trong `docs/Sample code/`: mỗi task là một script Keras chạy từ trên xuống dưới, không dùng class hay module chung.
