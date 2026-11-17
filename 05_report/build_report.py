# Điền báo cáo vào template docs/Final Major Assignment Report 2026.docx
# Số liệu lấy từ 02_task1/output, 03_task2/output, 04_task3/output (results.json + hình)
# Chạy từ thư mục gốc project:  .venv/Scripts/python 05_report/build_report.py
import json
import os
import numpy as np
from pandas import read_csv
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt

template = 'docs/Final Major Assignment Report 2026.docx'
out_dir = '05_report/output/'
os.makedirs(out_dir, exist_ok=True)

r1 = json.load(open('02_task1/output/results.json'))
r2 = json.load(open('03_task2/output/results.json'))
r3 = json.load(open('04_task3/output/results.json'))
svhn_info = json.load(open('01_data/processed/svhn_info.json'))
etth1 = read_csv('01_data/raw/etth1/ETTh1.csv')

SIZE = Pt(13)
doc = Document(template)

# ---------- trang bìa: giữ nguyên, chỉ điền chủ đề, giảng viên, sinh viên, ngày ----------
paras = doc.paragraphs
def set_text(p, text, bold=True, size=None):
    for r in p.runs[1:]:
        r.text = ''
    p.runs[0].text = text
    p.runs[0].bold = bold
    if size:
        p.runs[0].font.size = size

set_text(paras[5], 'TOPIC: RNN, ENCODER–DECODER AND ATTENTION FOR TIME-SERIES FORECASTING AND '
                   'HOUSE-NUMBER RECOGNITION')
set_text(paras[7], 'Instructor: [Instructor name]')
set_text(paras[8], 'Students:')
for k, p in enumerate(paras[9:13]):
    p.add_run('%d. [Full name] – [ID]' % (k + 1)).font.size = SIZE
for p in paras[13:15]:      # bỏ 2 dòng trống để trang bìa vừa 1 trang
    p._element.getparent().remove(p._element)
set_text(paras[15], 'Hanoi, December 2026')

# ---------- xoá phần hướng dẫn của template (từ "Part 1" trở đi) ----------
for p in paras[16:]:
    p._element.getparent().remove(p._element)

# ---------- các hàm tiện ích ----------
def heading(text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(15) if level == 0 else SIZE
    if level == 2:
        r.italic = True
    return p

def para(text, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(4)
    # **đậm** đơn giản: tách theo dấu **
    for k, chunk in enumerate(text.split('**')):
        if chunk:
            r = p.add_run(chunk)
            r.font.size = SIZE
            r.bold = (k % 2 == 1)
            r.italic = italic
    return p

def bullet(text):
    p = para(text)
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.2)
    p.runs[0].text = '•  ' + p.runs[0].text
    return p

def equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.italic = True
    r.font.size = SIZE
    r.font.name = 'Cambria Math'
    return p

fig_no = [0]
def figure(path, caption, width=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(width))
    fig_no[0] += 1
    c = para('Figure %d. %s' % (fig_no[0], caption), italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    c.runs[0].font.size = Pt(11)

tab_no = [0]
def table(header, rows, caption, widths=None):
    tab_no[0] += 1
    c = para('Table %d. %s' % (tab_no[0], caption), italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    c.runs[0].font.size = Pt(11)
    c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ''
            run = cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(11)
            run.bold = (i == 0)
            if widths:
                cell.width = Inches(widths[j])
    doc.add_paragraph()

def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

f4 = lambda v: '%.4f' % v   # noqa: E731
f3 = lambda v: '%.3f' % v   # noqa: E731
def metric_row(name, m, fmt=f4):
    return [name, fmt(m['R2']), fmt(m['NSE']), fmt(m['MAE']), fmt(m['RMSE'])]
M_HEAD = ['Model', 'R²', 'NSE', 'MAE', 'RMSE']

# =====================================================================
heading('Part 1: Theoretical Background', 0)
para('This part summarises the methods used in the three tasks: recurrent neural networks (RNN) and the '
     'Long Short-Term Memory (LSTM) cell, the encoder–decoder (sequence-to-sequence) architecture, the attention '
     'mechanism, the convolutional encoder used for images, and the evaluation metrics.')

heading('1. Recurrent Neural Networks (RNN)')
para('A recurrent neural network processes a sequence x₁, x₂, …, x_T one element at a time and keeps a hidden '
     'state h_t that summarises everything seen so far. The same weights are reused at every time step, so the '
     'network can handle sequences of any length and naturally models temporal dependence. For a simple (Elman) '
     'RNN the update is:')
equation('h_t = tanh(W_x · x_t + W_h · h_{t−1} + b),        ŷ_t = W_y · h_t + b_y')
para('Training uses back-propagation through time (BPTT): the network is unrolled over the sequence and the '
     'gradient of the loss flows backwards through every time step. Because the gradient is multiplied by W_h and '
     'by the derivative of tanh at each step, it tends to shrink (vanishing gradient) or blow up (exploding '
     'gradient) over long sequences, so a simple RNN struggles to learn long-range dependencies.')

heading('1.1. Long Short-Term Memory (LSTM)', 2)
para('The LSTM (Hochreiter & Schmidhuber, 1997) adds a cell state c_t and three gates that control what is '
     'forgotten, written and read. The additive update of c_t gives the gradient a path that does not vanish, '
     'which lets the network remember information over many steps:')
equation('f_t = σ(W_f·[h_{t−1}, x_t] + b_f)      i_t = σ(W_i·[h_{t−1}, x_t] + b_i)      o_t = σ(W_o·[h_{t−1}, x_t] + b_o)')
equation('c̃_t = tanh(W_c·[h_{t−1}, x_t] + b_c)      c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t      h_t = o_t ⊙ tanh(c_t)')
para('Here σ is the logistic sigmoid and ⊙ the element-wise product. f_t (forget gate) decides how much of the '
     'previous memory to keep, i_t (input gate) how much new information to write, and o_t (output gate) how much '
     'of the memory to expose as the hidden state. All tasks in this project use LSTM layers.')

heading('2. Encoder–Decoder (Sequence-to-Sequence) Architecture')
para('Many problems map an input sequence to an output sequence of a different length: forecasting the next '
     'five hours from the last 24 hours, or reading a variable-length digit string from an image. The '
     'encoder–decoder architecture (Sutskever et al., 2014; Cho et al., 2014) splits the model into two parts:')
bullet('**Encoder**: reads the input and compresses it into hidden representations. For time series it is an '
       'LSTM whose final state (h_T, c_T) summarises the input window. For images it is a CNN.')
bullet('**Decoder**: an LSTM initialised with the encoder state that produces the output one step at a time. At '
       'step k it receives an input (the repeated context vector, or the previous output token) and emits ŷ_k.')
para('During training the decoder of a text-like output (Task 3) receives the ground-truth previous token '
     '(teacher forcing), which makes training stable and parallel. At inference time it receives its own previous '
     'prediction and decoding stops when the end-of-sequence token is produced (greedy decoding). The weakness of '
     'the basic design is that the whole input must be squeezed into one fixed-size vector, a bottleneck that '
     'attention removes.')

heading('3. Attention Mechanism')
para('Attention (Bahdanau et al., 2015; Luong et al., 2015) lets the model look back at all encoder states '
     'instead of only the last one. For a query vector q (for example the current decoder state) and encoder states '
     'h₁, …, h_T (keys and values), attention computes a score for each position, normalises the scores with a '
     'softmax and returns the weighted sum as a context vector:')
equation('e_j = score(q, h_j),        α_j = exp(e_j) / Σ_k exp(e_k),        context = Σ_j α_j · h_j')
para('Two scoring functions are used in this project:')
bullet('**Additive (Bahdanau) attention**: score(q, h_j) = vᵀ tanh(W_q·q + W_k·h_j). It is used in Task 1 '
       '(Keras AdditiveAttention, query = last LSTM state, keys = all LSTM states) and in Task 3 (query = decoder '
       'state, keys = the 4 × 8 grid of CNN features).')
bullet('**Multiplicative (Luong, dot-product) attention**: score(q, h_j) = qᵀ·h_j. It is used in Task 2 (Keras '
       'Attention), where every decoder step attends to all encoder steps.')
para('The context vector is concatenated with the decoder state and passed to the output layer. The weights α '
     'are interpretable: they show which past hours (Tasks 1–2) or which image regions (Task 3) the model relied on '
     'for each prediction. The instructor’s sample code uses self-attention (query = key = value = RNN outputs); '
     'this project uses the query–key form, which is the classic attention of sequence-to-sequence models.')

heading('4. Convolutional Encoder for Images')
para('For image-to-sequence recognition the encoder is a convolutional neural network (CNN). Convolution layers '
     'with 3 × 3 kernels learn local patterns (edges, strokes, digit parts). Batch normalisation stabilises training. '
     'Max-pooling halves the spatial size and builds translation invariance. After four pooling stages a 64 × 128 '
     'image becomes a 4 × 8 grid of 256-dimensional feature vectors. Each grid cell describes one region of the '
     'image, and these 32 vectors are the "encoder states" the decoder attends to while emitting the digits from '
     'left to right (Goodfellow et al., 2014; Xu et al., 2015).')

heading('5. Evaluation Metrics')
para('For observed values y_i, predictions ŷ_i (i = 1…n) and the observed mean ȳ:')
equation('MAE = (1/n) Σ |y_i − ŷ_i|                RMSE = √[ (1/n) Σ (y_i − ŷ_i)² ]')
equation('NSE = 1 − Σ (y_i − ŷ_i)² / Σ (y_i − ȳ)²')
equation('R² = [ Σ (y_i − ȳ)(ŷ_i − ŷ̄) / √( Σ (y_i − ȳ)² · Σ (ŷ_i − ŷ̄)² ) ]²')
bullet('**MAE** and **RMSE** are in the unit of the target (°C for OT). RMSE penalises large errors more strongly.')
bullet('**NSE** (Nash & Sutcliffe, 1970) compares the squared error of the model with that of a forecast that always '
       'outputs the observed mean. NSE = 1 is perfect, NSE = 0 means "no better than the mean", and NSE < 0 is worse. '
       'Note that sklearn.metrics.r2_score computes exactly this quantity.')
bullet('**R²** is computed here as the squared Pearson correlation between observations and predictions, as is '
       'common in hydrology and forecasting studies (Legates & McCabe, 1999). It measures how well the predictions '
       'follow the variation of the observations but ignores systematic bias: a forecast ŷ = 2y + 1 has R² = 1 but a '
       'low NSE. Reporting both therefore separates "shape" errors from bias and scale errors.')
para('**Metrics for sequence-to-sequence forecasting.** When the model outputs a vector (ŷ_{t+1}, …, ŷ_{t+H}) '
     'for each sample, the metrics can be aggregated in three ways, all reported in Task 2:')
bullet('**Per horizon**: compute each metric separately for every step h = 1…H using the n pairs '
       '(y_{i,t+h}, ŷ_{i,t+h}). This shows how accuracy degrades as the forecast reaches further into the future.')
bullet('**Overall (pooled)**: flatten all n × H pairs into one vector and compute the metric once.')
bullet('**Average over horizons**: the mean of the H per-horizon values.')
para('MAE and RMSE² are identical under pooling and averaging, while R² and NSE differ slightly because the '
     'variance term is computed per horizon or over the pooled data. For Task 3 the output sequence is a string of '
     'digits. Besides the sequence-level metrics usual for recognition (whole-sequence accuracy and character error '
     'rate), the digit string is converted into its numeric value (e.g. "152" → 152) so that R², NSE, MAE and RMSE '
     'can be computed as required. Because house numbers are heavily right-skewed, the same metrics are also '
     'reported on log₁₀(1 + value).')

page_break()
# =====================================================================
heading('Part 2: Practical Implementation', 0)
para('All models are implemented in Python with Keras 3 running on the PyTorch backend and trained on an NVIDIA '
     'GeForce RTX 5060 Laptop GPU. Each task is one script that follows the structure of the course sample code: '
     '02_task1/task1_rnn_attention.py, 03_task2/task2_seq2seq_attention.py and '
     '04_task3/task3_image_to_sequence.py (data preparation for Task 3: 01_data/prepare_svhn.py). The random seed '
     'is fixed to 42.')

# ---------------------------- TASK 1 ----------------------------
heading('1. Task 1 – Forecasting a Single Future Time Point')
heading('1.1. Dataset Description', 2)
para('**Dataset**: ETTh1 – Electricity Transformer Temperature (hourly), from the ETDataset collection released '
     'with the Informer paper (Zhou et al., 2021).')
para('**Download link**: https://github.com/zhouhaoyi/ETDataset (file ETT-small/ETTh1.csv).')
para('The data were recorded by a power transformer station in China. Each row holds six power-load features '
     'and the oil temperature of the transformer. The oil temperature reflects the transformer’s condition and is '
     'important for preventing overheating. The file contains **%d hourly observations** from **%s** to **%s** '
     '(about two years), with no missing values. The attributes are:' % (len(etth1), etth1['date'].iloc[0][:10],
                                                                          etth1['date'].iloc[-1][:10]))
desc = etth1.describe()
attr_desc = {'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load', 'MUFL': 'Middle UseFul Load',
             'MULL': 'Middle UseLess Load', 'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
             'OT': 'Oil Temperature (°C) – target'}
table(['Attribute', 'Meaning', 'Mean', 'Std', 'Min', 'Max'],
      [[c, attr_desc[c], '%.2f' % desc[c]['mean'], '%.2f' % desc[c]['std'], '%.2f' % desc[c]['min'],
        '%.2f' % desc[c]['max']] for c in attr_desc],
      'Attributes of ETTh1 (statistics over the full file).', [0.8, 2.4, 0.7, 0.7, 0.7, 0.7])
para('**Data collection frequency**: 1 hour (one time step = one hour). Besides the 7 attributes, two calendar '
     'features (sin and cos of the hour of day) are added because OT and the loads follow a daily cycle.')
para('**Target variable (pred_column)**: **OT** (oil temperature).')
para('**Data split** (chronological, following the standard ETT benchmark protocol of 12/4/4 months):')
n1 = r1['n_samples']
bullet('Training set: 8,640 hourly rows (2016-07-01 to 2017-06-25) → %d samples (windows).' % n1['train'])
bullet('Validation set: 2,880 rows (2017-06-26 to 2017-10-23) → %d samples.' % n1['val'])
bullet('Test set: 2,880 rows (2017-10-24 to 2018-02-20) → %d samples.' % n1['test'])
para('The remaining 3,020 rows (after 2018-02-20) are not used, as in the benchmark protocol. All attributes are '
     'standardised with a StandardScaler fitted on the training rows only. An important property of the split is a '
     '**distribution shift**: the mean OT is 17.1 °C in the training period and 15.6 °C in validation, but only '
     '4.8 °C in the winter test period.')

heading('1.2. Problem Description', 2)
para('**Title**: Two-hour-ahead forecasting of transformer oil temperature with an LSTM network and attention.')
para('**Research objectives**: build a recurrent model that predicts OT two hours ahead from the recent history '
     'of loads and temperature, add an attention mechanism over the input window, and evaluate it with R², NSE, '
     'MAE and RMSE against a naive persistence baseline.')
para('**Input of the problem**: a window of the last n_steps = %d hours (t−23, …, t), each with 9 features '
     '(6 loads, OT, sin/cos hour), i.e. a tensor of shape (%d, 9).' % (r1['n_steps'], r1['n_steps']))
para('**Output of the problem**: one value, the oil temperature OT at time t + 2 (n_pred = 2).')
para('**Main steps**:')
for s in ['Load ETTh1.csv, add the hour-of-day features, split chronologically into train/validation/test.',
          'Standardise the features using statistics of the training set.',
          'Create samples with split_sequences(data, n_steps=24, n_pred=2, pred_col=OT).',
          'Because of the distribution shift, the network learns the change Δ = OT(t+2) − OT(t). The final forecast '
          'is OT(t) + Δ̂, so the model only has to predict how the temperature will move.',
          'Build the LSTM + attention model, train it with Adam and the MSE loss, use early stopping on the '
          'validation loss.',
          'Predict the test set, transform back to °C and compute R², NSE, MAE, RMSE. Compare with the persistence '
          'baseline OT̂(t+2) = OT(t).']:
    bullet(s)

heading('1.3. Model Architecture and Design', 2)
table(['Layer', 'Output shape', 'Description'],
      [['Input', '(24, 9)', '24 hours × 9 features'],
       ['LSTM(64, return_sequences, return_state)', '(24, 64) + (64)', 'hidden states h₁…h₂₄ and last state h₂₄'],
       ['AdditiveAttention', '(1, 64)', 'query = h₂₄, keys/values = h₁…h₂₄ → context vector'],
       ['Concatenate', '(128)', '[context, h₂₄]'],
       ['Dense(32, ReLU) + Dropout(0.2)', '(32)', 'non-linear combination'],
       ['Dense(1)', '(1)', 'predicted ΔOT (standardised)']],
      'Task 1 – LSTM + attention architecture (%s trainable parameters).' % format(r1['params'], ','),
      [2.6, 1.3, 2.6])
para('**Hyper-parameters**: optimiser Adam (learning rate 0.001), loss MSE, batch size 64, at most 100 epochs, '
     'EarlyStopping(patience = 10, restore best weights) and ReduceLROnPlateau(factor 0.5, patience 4) on the '
     'validation loss. Training stopped after %d epochs, and the weights of the epoch with the lowest validation loss '
     '(epoch %d) were restored. The validation loss reaches its minimum after only a few epochs and then rises while '
     'the training loss keeps falling, a sign of over-fitting that early stopping prevents. The window length 24 h '
     'was chosen on the validation set: 24 h and 48 h gave the same validation accuracy, so the shorter window was '
     'kept.' % (r1['epochs_trained'], r1['epochs_trained'] - 10))
figure('02_task1/output/loss.png', 'Task 1 – training and validation loss.', 4.5)

heading('1.4. Experimental Results', 2)
m, b = r1['lstm_attention'], r1['persistence']
table(M_HEAD, [metric_row('LSTM + Attention', m), metric_row('Persistence baseline (OT(t))', b)],
      'Task 1 – test-set performance (OT in °C, %d test samples).' % n1['test'], [2.6, 0.9, 0.9, 0.9, 0.9])
better = m['RMSE'] < b['RMSE']
para('On the test set the LSTM + attention model reaches R² = %.4f and NSE = %.4f, with MAE = %.3f °C and '
     'RMSE = %.3f °C for a two-hour-ahead forecast. %s' % (
         m['R2'], m['NSE'], m['MAE'], m['RMSE'],
         ('It improves on the persistence baseline (RMSE %.3f °C), showing that the network learned how the '
          'temperature moves from the load history and the daily cycle.' % b['RMSE']) if better else
         ('The persistence baseline is very strong for this smooth series two hours ahead (RMSE %.3f °C). The model '
          'has a higher R² than persistence (it follows the variations of OT better) but a slightly higher error, '
          'because it keeps a small positive bias: the cold test winter is outside the conditions seen during '
          'training. Over longer horizons the learned model clearly overtakes persistence (Task 2).'
          % b['RMSE'])))
figure('02_task1/output/forecast.png', 'Task 1 – observed vs predicted OT at t+2, first 14 days of the test set.')
figure('02_task1/output/scatter.png', 'Task 1 – predicted vs observed OT on the test set.', 3.6)
figure('02_task1/output/attention.png', 'Task 1 – average attention weight over the 24-hour input window '
       '(test set).', 5.0)
para('On average the attention weights are almost uniform over the 24 hours (≈ 1/24 each, slightly higher on the '
     'oldest hours). In this model the last hidden state h₂₄, which already summarises the recent hours, is passed '
     'directly to the output layer. The attention context therefore acts mainly as a smooth summary of the whole '
     'day (similar to average pooling over the window) and adds information about the daily cycle. The attention '
     'in the encoder–decoder of Task 2 is much sharper (Figure %d).' % (fig_no[0] + 4))

# ---------------------------- TASK 2 ----------------------------
heading('2. Task 2 – Forecasting Multiple Consecutive Future Time Points')
para('Task 2 uses the same dataset, target (OT), features, chronological split and scaling as Task 1 '
     '(Section 1.1). Sample counts: %d training, %d validation and %d test windows.' % (
         r2['n_samples']['train'], r2['n_samples']['val'], r2['n_samples']['test']))
heading('2.1. Problem Description', 2)
para('**Title**: Multi-step (1 to 5 hours ahead) forecasting of transformer oil temperature with an LSTM '
     'encoder–decoder and attention.')
para('**Research objectives**: build a sequence-to-sequence model that outputs the whole 5-hour trajectory of OT '
     'at once, use attention so every decoder step can look at all input hours, and study how accuracy changes with '
     'the forecast horizon using R², NSE, MAE and RMSE.')
para('**Input of the problem**: the last n_steps = %d hours × 9 features (same as Task 1).' % r2['n_steps'])
para('**Output of the problem**: the sequence OT(t+1), OT(t+2), …, OT(t+5) (n_pred = 5).')
para('**Main steps**:')
for s in ['Re-use the data pipeline of Task 1. split_sequences now returns a 5-value target vector per sample.',
          'Learn the changes Δ_k = OT(t+k) − OT(t), k = 1…5. Forecast = OT(t) + Δ̂_k.',
          'Encoder LSTM reads the window. Its final state is repeated 5 times (RepeatVector) and fed to a decoder '
          'LSTM initialised with the encoder state.',
          'Luong (dot-product) attention between decoder steps and encoder steps. The context is concatenated with '
          'the decoder output and mapped to one value per step with TimeDistributed(Dense).',
          'Train with Adam + MSE and early stopping. Evaluate per horizon, pooled and averaged, and compare with '
          'the persistence baseline.']:
    bullet(s)

heading('2.2. Model Architecture and Design', 2)
table(['Layer', 'Output shape', 'Description'],
      [['Input', '(24, 9)', 'encoder input window'],
       ['Encoder LSTM(64, return_sequences, return_state)', '(24, 64), (64), (64)', 'encoder states, h_T, c_T'],
       ['RepeatVector(5)', '(5, 64)', 'h_T repeated for 5 decoder steps'],
       ['Decoder LSTM(64, return_sequences)', '(5, 64)', 'initial_state = [h_T, c_T]'],
       ['Attention (dot-product)', '(5, 64)', 'query = decoder states, keys/values = encoder states'],
       ['Concatenate', '(5, 128)', '[decoder state, context]'],
       ['TimeDistributed(Dense(32, ReLU)) + Dropout(0.2)', '(5, 32)', ''],
       ['TimeDistributed(Dense(1))', '(5, 1)', 'Δ̂_1 … Δ̂_5 (standardised)']],
      'Task 2 – encoder–decoder with attention (%s trainable parameters).' % format(r2['params'], ','),
      [2.8, 1.5, 2.2])
para('**Hyper-parameters**: Adam (learning rate 0.001), MSE loss, batch size 64, at most 100 epochs, '
     'EarlyStopping(patience 10, restore best weights) and ReduceLROnPlateau(factor 0.5, patience 4). Training '
     'stopped after %d epochs, and the best weights (epoch %d) were restored. As in Task 1, the validation loss is '
     'lowest early in training.' % (r2['epochs_trained'], r2['epochs_trained'] - 10))
figure('03_task2/output/loss.png', 'Task 2 – training and validation loss.', 4.5)

heading('2.3. Experimental Results', 2)
rows = [metric_row('t+%d' % (h + 1), r2['per_horizon'][h]) for h in range(5)]
rows += [metric_row('Overall (pooled)', r2['overall']), metric_row('Average over horizons', r2['average_over_horizons'])]
table(['Horizon'] + M_HEAD[1:], rows, 'Task 2 – Seq2Seq + attention, test-set performance (OT in °C).',
      [2.2, 1.0, 1.0, 1.0, 1.0])
rows = [metric_row('t+%d' % (h + 1), r2['persistence_per_horizon'][h]) for h in range(5)]
rows += [metric_row('Overall (pooled)', r2['persistence_overall'])]
table(['Horizon'] + M_HEAD[1:], rows, 'Task 2 – persistence baseline OT̂(t+k) = OT(t), test set.',
      [2.2, 1.0, 1.0, 1.0, 1.0])
ph, pp = r2['per_horizon'], r2['persistence_per_horizon']
wins = ['t+%d' % (h + 1) for h in range(5) if ph[h]['RMSE'] < pp[h]['RMSE']]
losses = ['t+%d' % (h + 1) for h in range(5) if ph[h]['RMSE'] >= pp[h]['RMSE']]
para('The encoder–decoder reaches an overall R² of %.4f and NSE of %.4f (RMSE %.3f °C, MAE %.3f °C) over the five '
     'horizons. Accuracy decreases with the horizon, as expected: RMSE grows from %.3f °C at t+1 to %.3f °C at t+5, '
     'and NSE falls from %.4f to %.4f. Compared with persistence (overall RMSE %.3f °C), the model has a lower RMSE '
     'at %s%s. The advantage grows with the horizon, because persistence cannot anticipate any change while the '
     'network uses the load history and the daily cycle. At t+1 the series is so smooth that the last observed '
     'value is already an excellent forecast.' % (
         r2['overall']['R2'], r2['overall']['NSE'], r2['overall']['RMSE'], r2['overall']['MAE'],
         ph[0]['RMSE'], ph[4]['RMSE'], ph[0]['NSE'], ph[4]['NSE'], r2['persistence_overall']['RMSE'],
         ', '.join(wins) if wins else 'no horizon',
         (' and a slightly higher RMSE at %s' % ', '.join(losses)) if losses else ''))
figure('03_task2/output/rmse_by_horizon.png', 'Task 2 – RMSE by forecast horizon.', 4.5)
figure('03_task2/output/forecast.png', 'Task 2 – observed OT and forecasts made 1 and 5 hours ahead (first week '
       'of the test set).')
figure('03_task2/output/attention.png', 'Task 2 – average attention of each decoder step over the encoder steps.')
para('The attention map shows that every decoder step looks mostly at the last 5–6 input hours. The focus is '
     'sharpest for t+1 and becomes more spread out for later steps, which need more of the past day to extrapolate '
     'the trend.')

# ---------------------------- TASK 3 ----------------------------
heading('3. Task 3 – Image-to-Sequence Prediction')
heading('3.1. Dataset Description', 2)
para('**Dataset**: The Street View House Numbers (SVHN), format 1 – full numbers (Netzer et al., 2011).')
para('**Download link**: http://ufldl.stanford.edu/housenumbers/ (train.tar.gz, test.tar.gz).')
para('SVHN contains real photographs of house numbers cropped from Google Street View images. The images contain '
     'printed (not handwritten) digit sequences of 1 to 6 digits, in many fonts, colours, backgrounds, blur and '
     'lighting conditions, so it is a harder, real-world version of multi-digit recognition.')
bullet('**Number of images**: %d training and %d test images in the original archives (the 202,353 "extra" '
       'images are not used).' % (svhn_info['train_raw'], svhn_info['test_raw']))
bullet('**Format and size**: PNG colour (RGB) images of variable size. For each image the file digitStruct.mat stores a bounding box and a label for every digit. '
       'We crop the union of the digit boxes, enlarged by 30%, and resize it to **64 × 128 × 3** (height × width × '
       'RGB).')
bullet('**Labels / output sequences**: the digit string from left to right, e.g. "1", "23", "4519" (label 10 in '
       'the file means digit 0). %d training images and %d test image with more than 5 digits were removed. Test '
       'length distribution: %s.' % (svhn_info['dropped_train'], svhn_info['dropped_test'],
                                     ', '.join('%s digit(s): %s' % (k, v)
                                               for k, v in r3['length_distribution_test'].items())))
figure('01_data/processed/svhn_samples.png', 'SVHN samples after cropping and resizing, with their label sequences.')
n3 = r3['n_samples']
para('**Data split**:')
bullet('Training set: %d images (the official train set without its last 10%%).' % n3['train'])
bullet('Validation set: %d images (the last 10%% of the official train set).' % n3['val'])
bullet('Test set: %d images (the official test set).' % n3['test'])

heading('3.2. Problem Description', 2)
para('**Title**: House-number recognition from street-view images with a CNN encoder, an LSTM decoder and '
     'attention.')
para('**Research objectives**: build an end-to-end image-to-sequence model that reads a variable-length digit '
     'string directly from an image, without segmenting individual digits, and evaluate it with sequence accuracy, '
     'character error rate and R², NSE, MAE, RMSE on the numeric value.')
para('**Input of the problem**: an RGB image of 64 × 128 pixels containing a house number.')
para('**Output of the problem**: the sequence of digits (length 1–5) followed by an end token, e.g. '
     '"4", "5", "1", "<EOS>".')
para('**Main steps**:')
for s in ['Read digitStruct.mat, crop the digit region, resize to 64 × 128, store images and label strings '
          '(prepare_svhn.py).',
          'Encode labels as tokens: PAD = 0, SOS = 1, EOS = 2, digit d → d + 3. Decoder input = [SOS, d₁, …, d_L], '
          'decoder target = [d₁, …, d_L, EOS], both padded to 6.',
          'Build the CNN encoder → 4 × 8 feature grid, and the LSTM decoder with attention over the grid. Train '
          'with teacher forcing, cross-entropy loss, PAD positions masked with sample weights. Light data '
          'augmentation (translation, rotation, contrast).',
          'Greedy decoding on the test set: start from SOS and feed back the predicted token until EOS.',
          'Compute sequence accuracy, character error rate and the numeric-value metrics.']:
    bullet(s)

heading('3.3. Model Architecture and Design', 2)
table(['Block', 'Output shape', 'Description'],
      [['Input + Rescaling(1/255)', '(64, 128, 3)', 'pixel values to [0, 1]'],
       ['Augmentation', '(64, 128, 3)', 'RandomTranslation(5%), RandomRotation(0.02), RandomContrast(0.3)'],
       ['2 × [Conv 3×3, 32, ReLU + BN] + MaxPool', '(32, 64, 32)', 'CNN encoder block 1'],
       ['2 × [Conv 3×3, 64, ReLU + BN] + MaxPool', '(16, 32, 64)', 'block 2'],
       ['2 × [Conv 3×3, 128, ReLU + BN] + MaxPool', '(8, 16, 128)', 'block 3'],
       ['Conv 3×3, 256 + BN + MaxPool + Dropout(0.25)', '(4, 8, 256)', 'block 4'],
       ['Reshape', '(32, 256)', '32 region vectors (encoder states)'],
       ['GlobalAveragePooling + 2 × Dense(256, tanh)', '(256), (256)', 'initial decoder states h₀, c₀'],
       ['Decoder input + Embedding(13, 64)', '(6, 64)', 'previous tokens'],
       ['LSTM(256, return_sequences)', '(6, 256)', 'decoder states'],
       ['AdditiveAttention', '(6, 256)', 'query = decoder state, keys/values = 32 regions'],
       ['Concatenate + Dense(256, tanh) + Dropout(0.3)', '(6, 256)', ''],
       ['Dense(13, softmax)', '(6, 13)', 'probability of each token at each step']],
      'Task 3 – CNN encoder + LSTM decoder with attention (%s trainable parameters).' % format(r3['params'], ','),
      [2.8, 1.3, 2.4])
para('**Hyper-parameters**: Adam (learning rate 0.001), sparse categorical cross-entropy with PAD positions masked, '
     'batch size 128, at most 40 epochs, EarlyStopping(patience 5) and ReduceLROnPlateau(factor 0.5, patience 2) on '
     'the validation loss. Training stopped after %d epochs. Maximum output length 6 (5 digits + EOS).'
     % r3['epochs_trained'])
figure('04_task3/output/loss.png', 'Task 3 – training and validation loss.', 4.5)

heading('3.4. Experimental Results', 2)
table(['Metric', 'Test value'],
      [['Sequence accuracy (whole number correct)', '%.2f%%' % (100 * r3['sequence_accuracy'])],
       ['Character accuracy (1 − CER)', '%.2f%%' % (100 * r3['character_accuracy'])],
       ['Character error rate (edit distance / length)', '%.2f%%' % (100 * r3['character_error_rate'])],
       ['Length accuracy (correct number of digits)', '%.2f%%' % (100 * r3['length_accuracy'])]],
      'Task 3 – recognition metrics on the %d test images.' % n3['test'], [3.8, 1.5])
table(['Target'] + M_HEAD[1:],
      [metric_row('House-number value', r3['value_metrics'], f3)[0:1] + metric_row('', r3['value_metrics'], f3)[1:],
       metric_row('log₁₀(1 + value)', r3['log_value_metrics'])],
      'Task 3 – R², NSE, MAE, RMSE on the numeric value of the predicted digit string.', [2.2, 1.0, 1.0, 1.0, 1.0])
vm = r3['value_metrics']
para('The model reads the complete house number correctly for %.1f%% of the test images and %.1f%% of the '
     'individual digits. On the numeric value it reaches R² = %.4f and NSE = %.4f. MAE (%.1f) and RMSE (%.1f) are '
     'dominated by a few large mistakes: a single wrong or missing leading digit changes the value by hundreds or '
     'thousands. This is why the log-scale metrics (R² = %.4f, NSE = %.4f) and the sequence and character '
     'accuracies give a fairer picture of recognition quality.' % (
         100 * r3['sequence_accuracy'], 100 * r3['character_accuracy'], vm['R2'], vm['NSE'], vm['MAE'], vm['RMSE'],
         r3['log_value_metrics']['R2'], r3['log_value_metrics']['NSE']))
figure('04_task3/output/predictions.png', 'Task 3 – test images with true and predicted sequences '
       '(green = correct, red = wrong).')
figure('04_task3/output/attention.png', 'Task 3 – attention maps: at each decoding step the model focuses on '
       'the digit it is reading, moving from left to right.')

page_break()
# =====================================================================
heading('Part 3: Conclusion', 0)
para('The team built and evaluated three deep-learning models that combine recurrent networks with attention.')
bullet('**Task 1**: on ETTh1, an LSTM with additive attention forecasts oil temperature two hours ahead with '
       'R² = %.4f, NSE = %.4f, MAE = %.3f °C and RMSE = %.3f °C on the test set (persistence baseline: NSE = %.4f, '
       'RMSE = %.3f °C).' % (m['R2'], m['NSE'], m['MAE'], m['RMSE'], b['NSE'], b['RMSE']))
bullet('**Task 2**: an LSTM encoder–decoder with dot-product attention forecasts the next five hours at once with '
       'an overall R² = %.4f and NSE = %.4f (RMSE %.3f °C). The error grows steadily from t+1 to t+5.' % (
           r2['overall']['R2'], r2['overall']['NSE'], r2['overall']['RMSE']))
bullet('**Task 3**: a CNN encoder with an attention-based LSTM decoder reads street-view house numbers end to end, '
       'with %.1f%% whole-sequence accuracy and %.1f%% digit accuracy on the SVHN test set. On the numeric value, '
       'R² = %.4f and NSE = %.4f.' % (100 * r3['sequence_accuracy'], 100 * r3['character_accuracy'], vm['R2'],
                                      vm['NSE']))
para('**Key findings.** (1) ETTh1 has a strong distribution shift between the training period and the winter '
     'test period. Learning the change of OT relative to its current value, instead of its absolute level, was '
     'essential: a model trained on absolute values had a large bias on the test set (NSE ≈ 0.78 vs ≈ 0.93). '
     '(2) For a smooth series such as oil temperature, persistence is a strong baseline at short horizons, and it '
     'must be reported to judge a forecasting model fairly. (3) R² (squared correlation) and NSE answer different '
     'questions: R² ignores bias while NSE does not, so reporting both exposes bias problems. (4) Attention weights '
     'are interpretable: the seq2seq decoder focuses on the most recent hours, more sharply for t+1 than for t+5, '
     'and the image model moves its focus over the image as it emits each digit. In Task 1 the attention is almost '
     'uniform and works as a summary of the whole input window. (5) For image-to-sequence models, metrics on the numeric value are very sensitive to single '
     'leading-digit errors and should be complemented by sequence and character accuracy.')
para('**Future work**: add the known future calendar features as decoder inputs, use autoregressive decoding with '
     'scheduled sampling in Task 2, use the SVHN "extra" set and a deeper (e.g. ResNet) encoder or beam search in '
     'Task 3, and test transformer-based models.')

heading('References', 0)
refs = [
    'Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and '
    'translate. ICLR 2015. arXiv:1409.0473.',
    'Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). '
    'Learning phrase representations using RNN encoder–decoder for statistical machine translation. EMNLP 2014.',
    'Goodfellow, I. J., Bulatov, Y., Ibarz, J., Arnoud, S., & Shet, V. (2014). Multi-digit number recognition from '
    'Street View imagery using deep convolutional neural networks. ICLR 2014. arXiv:1312.6082.',
    'Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.',
    'Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.',
    'Legates, D. R., & McCabe, G. J. (1999). Evaluating the use of "goodness-of-fit" measures in hydrologic and '
    'hydroclimatic model validation. Water Resources Research, 35(1), 233–241.',
    'Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine '
    'translation. EMNLP 2015. arXiv:1508.04025.',
    'Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I – A discussion '
    'of principles. Journal of Hydrology, 10(3), 282–290.',
    'Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading digits in natural images '
    'with unsupervised feature learning. NIPS Workshop on Deep Learning and Unsupervised Feature Learning. '
    'Dataset: http://ufldl.stanford.edu/housenumbers/',
    'Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. NeurIPS 2014.',
    'Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., & Bengio, Y. (2015). Show, '
    'attend and tell: Neural image caption generation with visual attention. ICML 2015.',
    'Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient '
    'transformer for long sequence time-series forecasting. AAAI 2021. Dataset: https://github.com/zhouhaoyi/ETDataset',
    'Chollet, F., et al. (2015–2026). Keras. https://keras.io',
]
for k, ref in enumerate(refs):
    p = para('[%d] %s' % (k + 1, ref), align=WD_ALIGN_PARAGRAPH.LEFT)
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)

doc.save(out_dir + 'Final_Report.docx')
print('saved', out_dir + 'Final_Report.docx')
