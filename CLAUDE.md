# DeepLearning — NEU Deep Learning final project (2026)

Group final project for the Deep Learning course. Three tasks, one report:
1. RNN + attention forecast of ETTh1 `OT` at t+2.
2. Encoder–decoder + attention forecast of `OT` at t+1…t+5.
3. Image→sequence on SVHN house numbers (CNN encoder + LSTM decoder + attention).
Every task reports R², NSE, MAE, RMSE.

Pipeline and gates: `CONTEXT.md`.

## Where things live

| Path | What it holds |
|---|---|
| `docs/` | the brief (PDF), the report template (.docx), the instructor's sample code. Reference only, never edit |
| `01_data/` | `raw/` downloads, `processed/` prepared arrays, `prepare_svhn.py` |
| `02_task1/` | `task1_rnn_attention.py`, `output/` (results.json, figures) |
| `03_task2/` | `task2_seq2seq_attention.py`, `output/` |
| `04_task3/` | `task3_image_to_sequence.py`, `output/` |
| `05_report/` | `build_report.py` → `output/Final_Report.docx` |
| `.venv/` | Python 3.14 venv (system torch + keras, sklearn, h5py, python-docx) |

## Rules

- **Code style must match `docs/Sample code/`.** Each task is one flat top-to-bottom Keras script: `split_sequences(...)`, `n_steps`/`n_pred`/`pred_col`, `Input → layers → Model`, compile/fit/predict, then print the metrics. No classes, no shared modules.
- Keras 3 runs on the torch backend: every script sets `os.environ["KERAS_BACKEND"] = "torch"` before importing keras.
- Run every script from the project root: `PYTHONUTF8=1 .venv/Scripts/python <stage>/<script>.py`. Without `PYTHONUTF8=1`, the Vietnamese `print`s crash once output is redirected (cp1252).
- R² = squared Pearson r, NSE = 1 − SSE/SST. They are deliberately different, and the report explains why.
- Task 1/2 predict the change in OT relative to OT(t), because the ETTh1 test period is much colder than train. Keep this.
- The report is in English. Numbers in it come from each `output/results.json`, never typed by hand.
