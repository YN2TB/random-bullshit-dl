# Task 2: Encoder-Decoder (Seq2Seq) + Attention dự báo OT cho 5 giờ tiếp theo (t+1 ... t+5), ETTh1
# Chạy từ thư mục gốc project:  .venv/Scripts/python 03_task2/task2_seq2seq_attention.py
import os
os.environ["KERAS_BACKEND"] = "torch"   # Keras 3 chạy trên PyTorch (GPU)

import json
from numpy import sqrt, asarray, mean, corrcoef, sin, cos, pi, repeat
from pandas import read_csv, to_datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Attention, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import set_random_seed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

set_random_seed(42)
out_dir = '03_task2/output/'
os.makedirs(out_dir, exist_ok=True)

# split a multivariate sequence into samples: input n_steps giờ, output n_pred giờ tiếp theo
def split_sequences(data, n_steps, n_pred, pred_col):
    X, y = [], []
    for i in range(len(data) - n_steps - n_pred + 1):
        X.append(data[i:i+n_steps, :])
        y.append(data[i+n_steps:i+n_steps+n_pred, pred_col])
    return asarray(X), asarray(y)

# các chỉ số đánh giá
def nse(y_true, y_pred):
    return 1 - sum((y_true - y_pred)**2) / sum((y_true - mean(y_true))**2)

def r2(y_true, y_pred):
    # R2 = bình phương hệ số tương quan Pearson giữa giá trị thực và dự báo
    return corrcoef(y_true, y_pred)[0, 1]**2

def evaluate(y_true, y_pred):
    return {'R2': r2(y_true, y_pred), 'NSE': nse(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred), 'RMSE': sqrt(mean_squared_error(y_true, y_pred))}

# 1. Đọc dữ liệu
filename = '01_data/raw/etth1/ETTh1.csv'
all_attributes = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
df = read_csv(filename, encoding='utf-8')
# thêm giờ trong ngày dạng sin/cos (OT có chu kỳ theo ngày)
hour = to_datetime(df['date']).dt.hour
df['hour_sin'] = sin(2 * pi * hour / 24)
df['hour_cos'] = cos(2 * pi * hour / 24)
all_attributes = all_attributes + ['hour_sin', 'hour_cos']
data = df[all_attributes].values.astype('float32')
print(df)

# 2. Chia train / val / test theo thời gian: 12 tháng / 4 tháng / 4 tháng (giống Task 1)
n_train = 12 * 30 * 24
n_val = 4 * 30 * 24
n_test = 4 * 30 * 24
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:n_train + n_val + n_test]
print("Train:", train_data.shape, " Val:", val_data.shape, " Test:", test_data.shape)

# 3. Chuẩn hoá (fit scaler trên tập train)
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# 4. Tạo mẫu: 24 giờ trước -> OT của 5 giờ tiếp theo
n_steps = 24
n_pred = 5
pred_col = all_attributes.index('OT')

X_train, y_train = split_sequences(train_data, n_steps, n_pred, pred_col)
X_val, y_val = split_sequences(val_data, n_steps, n_pred, pred_col)
X_test, y_test = split_sequences(test_data, n_steps, n_pred, pred_col)
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:", X_val.shape, " X_test:", X_test.shape)
print("X0:", X_train[0])
print("y0:", y_train[0])

# Giống Task 1: mô hình học độ thay đổi so với OT hiện tại, delta_k = OT(t+k) - OT(t)
y_train_delta = y_train - X_train[:, -1:, pred_col]
y_val_delta = y_val - X_val[:, -1:, pred_col]
y_train_delta = y_train_delta.reshape(y_train_delta.shape[0], n_pred, 1)
y_val_delta = y_val_delta.reshape(y_val_delta.shape[0], n_pred, 1)

# 5. Xây dựng mô hình Encoder-Decoder + Attention
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Encoder: LSTM đọc chuỗi đầu vào, trả về toàn bộ hidden state và trạng thái cuối
encoder_outputs, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(inputs)

# Decoder: lặp context vector n_pred lần, LSTM khởi tạo bằng trạng thái của encoder
decoder_inputs = RepeatVector(n_pred)(state_h)
decoder_outputs = LSTM(64, return_sequences=True)(decoder_inputs, initial_state=[state_h, state_c])

# Attention (Luong): mỗi bước của decoder nhìn lại toàn bộ các bước của encoder
attention_output, attention_scores = Attention()([decoder_outputs, encoder_outputs], return_attention_scores=True)
x = Concatenate()([decoder_outputs, attention_output])
x = TimeDistributed(Dense(32, activation='relu'))(x)
x = Dropout(0.2)(x)
outputs = TimeDistributed(Dense(1))(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
attention_model = Model(inputs=inputs, outputs=attention_scores)

# 6. Compile và train
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
history = model.fit(X_train, y_train_delta, epochs=100, batch_size=64, verbose=2,
                    validation_data=(X_val, y_val_delta), callbacks=[early_stop, reduce_lr])

# 7. Dự báo và đưa về đơn vị gốc (°C)
y_pred = X_test[:, -1:, pred_col] + model.predict(X_test).reshape(len(X_test), n_pred)
ot_mean, ot_std = scaler.mean_[pred_col], scaler.scale_[pred_col]
y_test_inv = y_test * ot_std + ot_mean
y_pred_inv = y_pred * ot_std + ot_mean
print("Dự báo đầu tiên:", y_pred_inv[0])
print("Thực tế đầu tiên:", y_test_inv[0])

# 8. Đánh giá trên tập test
# (a) theo từng bước dự báo t+1 ... t+5
# (b) gộp toàn bộ (mẫu, bước) thành một vector
# (c) trung bình của các chỉ số theo từng bước
y_naive = repeat(X_test[:, -1:, pred_col] * ot_std + ot_mean, n_pred, axis=1)   # baseline: OT(t) cho mọi bước
per_horizon, per_horizon_naive = [], []
for h in range(n_pred):
    per_horizon.append(evaluate(y_test_inv[:, h], y_pred_inv[:, h]))
    per_horizon_naive.append(evaluate(y_test_inv[:, h], y_naive[:, h]))
    print(f"t+{h + 1}:", per_horizon[h], "  persistence:", per_horizon_naive[h])
overall = evaluate(y_test_inv.reshape(-1), y_pred_inv.reshape(-1))
overall_naive = evaluate(y_test_inv.reshape(-1), y_naive.reshape(-1))
average = {k: mean([m[k] for m in per_horizon]) for k in overall}
print("Overall (flatten):", overall)
print("Average over horizons:", average)
print("Persistence overall:", overall_naive)

# 9. Lưu kết quả và hình
with open(out_dir + 'results.json', 'w') as f:
    json.dump({'n_steps': n_steps, 'n_pred': n_pred,
               'n_samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
               'epochs_trained': len(history.history['loss']), 'params': model.count_params(),
               'per_horizon': per_horizon, 'overall': overall, 'average_over_horizons': average,
               'persistence_per_horizon': per_horizon_naive, 'persistence_overall': overall_naive},
              f, indent=2, default=float)

plt.figure(figsize=(6, 3.5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch'); plt.ylabel('MSE (scaled)'); plt.title('Task 2 - learning curve'); plt.legend()
plt.tight_layout(); plt.savefig(out_dir + 'loss.png', dpi=150); plt.close()

plt.figure(figsize=(6, 3.5))
steps = range(1, n_pred + 1)
plt.plot(steps, [m['RMSE'] for m in per_horizon], 'o-', label='Seq2Seq + Attention')
plt.plot(steps, [m['RMSE'] for m in per_horizon_naive], 's--', label='persistence')
plt.xlabel('forecast horizon (hours ahead)'); plt.ylabel('RMSE (°C)'); plt.title('Task 2 - error by horizon')
plt.xticks(list(steps)); plt.legend()
plt.tight_layout(); plt.savefig(out_dir + 'rmse_by_horizon.png', dpi=150); plt.close()

n_show = 24 * 7
plt.figure(figsize=(10, 3.5))
plt.plot(y_test_inv[:n_show, 0], label='observed')
plt.plot(y_pred_inv[:n_show, 0], label='forecast t+1')
plt.plot(range(4, n_show + 4), y_pred_inv[:n_show, 4], label='forecast t+5', alpha=0.8)
plt.xlabel('hour (test set, first 7 days)'); plt.ylabel('OT (°C)'); plt.title('Task 2 - forecasts'); plt.legend()
plt.tight_layout(); plt.savefig(out_dir + 'forecast.png', dpi=150); plt.close()

weights = attention_model.predict(X_test).mean(axis=0)     # (n_pred, n_steps)
plt.figure(figsize=(8, 2.8))
plt.imshow(weights, aspect='auto', cmap='viridis', extent=[-n_steps + 0.5, 0.5, n_pred + 0.5, 0.5])
plt.colorbar(label='mean attention weight')
plt.xlabel('encoder step (hour relative to t)'); plt.ylabel('decoder step (t+k)')
plt.title('Task 2 - average attention weights (test)')
plt.tight_layout(); plt.savefig(out_dir + 'attention.png', dpi=150); plt.close()
