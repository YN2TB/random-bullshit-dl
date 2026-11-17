# Task 1: RNN + Attention dự báo OT tại thời điểm t+2 (ETTh1, mỗi bước = 1 giờ)
# Chạy từ thư mục gốc project:  .venv/Scripts/python 02_task1/task1_rnn_attention.py
import os
os.environ["KERAS_BACKEND"] = "torch"   # Keras 3 chạy trên PyTorch (GPU)

import json
from numpy import sqrt, asarray, mean, corrcoef, sin, cos, pi
from pandas import read_csv, to_datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import Input, LSTM, Dense, Dropout, AdditiveAttention, Reshape, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import set_random_seed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

set_random_seed(42)
out_dir = '02_task1/output/'
os.makedirs(out_dir, exist_ok=True)

# split a multivariate sequence into samples
def split_sequences(data, n_steps, n_pred, pred_col):
    X, y = [], []
    for i in range(len(data) - n_steps - n_pred + 1):
        # input: n_steps giờ gần nhất (t-n_steps+1 .. t), output: giá trị tại t+n_pred
        X.append(data[i:i+n_steps, :])
        y.append(data[i+n_steps+n_pred-1, pred_col])
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
dates = df['date'].values
# thêm giờ trong ngày dạng sin/cos (OT có chu kỳ theo ngày)
hour = to_datetime(df['date']).dt.hour
df['hour_sin'] = sin(2 * pi * hour / 24)
df['hour_cos'] = cos(2 * pi * hour / 24)
all_attributes = all_attributes + ['hour_sin', 'hour_cos']
data = df[all_attributes].values.astype('float32')
print(df)

# 2. Chia train / val / test theo thời gian: 12 tháng / 4 tháng / 4 tháng (chuẩn benchmark ETT)
n_train = 12 * 30 * 24
n_val = 4 * 30 * 24
n_test = 4 * 30 * 24
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:n_train + n_val + n_test]
test_dates = dates[n_train + n_val:n_train + n_val + n_test]
print("Train:", train_data.shape, " Val:", val_data.shape, " Test:", test_data.shape)

# 3. Chuẩn hoá (fit scaler trên tập train)
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# 4. Tạo mẫu: dùng 24 giờ trước để dự báo OT sau 2 giờ
n_steps = 24
n_pred = 2
pred_col = all_attributes.index('OT')

X_train, y_train = split_sequences(train_data, n_steps, n_pred, pred_col)
X_val, y_val = split_sequences(val_data, n_steps, n_pred, pred_col)
X_test, y_test = split_sequences(test_data, n_steps, n_pred, pred_col)
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:", X_val.shape, " X_test:", X_test.shape)
print("X0:", X_train[0])
print("y0:", y_train[0])

# Tập test (mùa đông 2017-2018) có OT thấp hơn hẳn tập train, nên mô hình học mức tuyệt đối sẽ bị lệch.
# Vì vậy mô hình học độ thay đổi: delta = OT(t+2) - OT(t), sau đó dự báo OT(t+2) = OT(t) + delta.
y_train_delta = y_train - X_train[:, -1, pred_col]
y_val_delta = y_val - X_val[:, -1, pred_col]

# 5. Xây dựng mô hình LSTM + Attention
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM trả về toàn bộ chuỗi hidden state và hidden state cuối cùng
lstm_output, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(inputs)

# Attention (Bahdanau): query = hidden state cuối, key = value = các hidden state
query = Reshape((1, 64))(state_h)
context, attention_scores = AdditiveAttention()([query, lstm_output], return_attention_scores=True)
context = Reshape((64,))(context)

# Ghép context vector với hidden state cuối rồi đưa qua Dense
x = Concatenate()([context, state_h])
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
attention_model = Model(inputs=inputs, outputs=attention_scores)   # dùng để xem trọng số attention

# 6. Compile và train
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
history = model.fit(X_train, y_train_delta, epochs=100, batch_size=64, verbose=2,
                    validation_data=(X_val, y_val_delta), callbacks=[early_stop, reduce_lr])

# 7. Dự báo và đưa về đơn vị gốc (°C)
y_pred = X_test[:, -1, pred_col] + model.predict(X_test).reshape(-1)
ot_mean, ot_std = scaler.mean_[pred_col], scaler.scale_[pred_col]
y_test_inv = y_test * ot_std + ot_mean
y_pred_inv = y_pred * ot_std + ot_mean
print("y_test=", y_test_inv[:10])
print("y_pred=", y_pred_inv[:10])

# 8. Đánh giá trên tập test
result = evaluate(y_test_inv, y_pred_inv)
print('LSTM + Attention:', result)

# baseline: giá trị OT hiện tại (thời điểm t) dùng làm dự báo cho t+2
y_naive = X_test[:, -1, pred_col] * ot_std + ot_mean
result_naive = evaluate(y_test_inv, y_naive)
print('Persistence baseline:', result_naive)

# 9. Lưu kết quả và hình
with open(out_dir + 'results.json', 'w') as f:
    json.dump({'n_steps': n_steps, 'n_pred': n_pred,
               'n_samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
               'epochs_trained': len(history.history['loss']), 'params': model.count_params(),
               'lstm_attention': result, 'persistence': result_naive}, f, indent=2, default=float)

plt.figure(figsize=(6, 3.5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch'); plt.ylabel('MSE (scaled)'); plt.title('Task 1 - learning curve'); plt.legend()
plt.tight_layout(); plt.savefig(out_dir + 'loss.png', dpi=150); plt.close()

n_show = 24 * 14
plt.figure(figsize=(10, 3.5))
plt.plot(y_test_inv[:n_show], label='observed')
plt.plot(y_pred_inv[:n_show], label='LSTM + Attention')
plt.xlabel('hour (test set, first 14 days)'); plt.ylabel('OT (°C)'); plt.title('Task 1 - OT at t+2'); plt.legend()
plt.tight_layout(); plt.savefig(out_dir + 'forecast.png', dpi=150); plt.close()

plt.figure(figsize=(4.5, 4.5))
plt.scatter(y_test_inv, y_pred_inv, s=3, alpha=0.4)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'k--')
plt.xlabel('observed OT'); plt.ylabel('predicted OT'); plt.title('Task 1 - test set')
plt.tight_layout(); plt.savefig(out_dir + 'scatter.png', dpi=150); plt.close()

weights = attention_model.predict(X_test).reshape(len(X_test), n_steps).mean(axis=0)
plt.figure(figsize=(7, 3))
plt.bar(range(-n_steps + 1, 1), weights)
plt.xlabel('hour relative to t'); plt.ylabel('mean attention weight'); plt.title('Task 1 - attention over the input window')
plt.tight_layout(); plt.savefig(out_dir + 'attention.png', dpi=150); plt.close()
