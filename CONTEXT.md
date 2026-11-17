# Pipeline

| Stage | Reads | Runs | Writes | Human check |
|---|---|---|---|---|
| 01_data | internet | download ETTh1 CSV + SVHN tar.gz into `raw/`, then `prepare_svhn.py` | `processed/svhn_64x128.npz`, `svhn_samples.png` | sample crops match their labels |
| 02_task1 | `01_data/raw/etth1/ETTh1.csv` | `task1_rnn_attention.py` | `output/results.json`, `loss/forecast/scatter/attention.png` | model vs persistence baseline |
| 03_task2 | same CSV | `task2_seq2seq_attention.py` | `output/results.json`, figures | error grows with horizon; beats baseline |
| 04_task3 | `01_data/processed/svhn_64x128.npz` | `task3_image_to_sequence.py` | `output/results.json`, figures | sequence accuracy, sample predictions |
| 05_report | `docs/*.docx` template, every `output/` | `build_report.py` | `05_report/output/Final_Report.docx` | read the whole report and fill in names |

Download commands (from the project root, then run `prepare_svhn.py`):
`curl -L -o 01_data/raw/etth1/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`
`curl -L -o 01_data/raw/svhn/train.tar.gz http://ufldl.stanford.edu/housenumbers/train.tar.gz` (same for `test.tar.gz`). `prepare_svhn.py` extracts them itself on the first run.

Run one script at a time. Task 3 takes ~65 min on the RTX 5060 and needs several GB of free RAM: two jobs together were killed for low memory.

Data sources:
- ETTh1: https://github.com/zhouhaoyi/ETDataset (`ETT-small/ETTh1.csv`)
- SVHN format 1: http://ufldl.stanford.edu/housenumbers/ (`train.tar.gz`, `test.tar.gz`, extracted in `01_data/raw/svhn/`)

Splits:
- ETTh1: 12 / 4 / 4 months (8640 / 2880 / 2880 rows).
- SVHN: official train with the last 10% held out as validation; official test as test.
