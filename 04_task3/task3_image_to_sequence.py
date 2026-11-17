# Task 3: Image-to-Sequence - nhận dạng dãy chữ số số nhà trong ảnh SVHN
# CNN Encoder + LSTM Decoder + Attention (Bahdanau)
# Chạy từ thư mục gốc project (sau khi chạy 01_data/prepare_svhn.py):
#   .venv/Scripts/python 04_task3/task3_image_to_sequence.py
import os
os.environ["KERAS_BACKEND"] = "torch"   # Keras 3 chạy trên PyTorch (GPU)

import json
import numpy as np
from numpy import sqrt, mean, corrcoef, log10
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import (Input, Rescaling, RandomTranslation, RandomRotation, RandomContrast, Conv2D,
                          BatchNormalization, MaxPooling2D, Dropout, Reshape, GlobalAveragePooling1D, Dense,
                          Embedding, LSTM, AdditiveAttention, Concatenate)
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import set_random_seed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

set_random_seed(42)
out_dir = '04_task3/output/'
os.makedirs(out_dir, exist_ok=True)

# 1. Load dữ liệu SVHN đã cắt sẵn (64 x 128 x 3) và nhãn dạng chuỗi, ví dụ '123'
data = np.load('01_data/processed/svhn_64x128.npz')
x_train, labels_train = data['X_train'], data['labels_train']
x_test, labels_test = data['X_test'], data['labels_test']

# Tách validation set từ training set (10% cuối)
n_val = len(x_train) // 10
x_val, labels_val = x_train[-n_val:], labels_train[-n_val:]
x_train, labels_train = x_train[:-n_val], labels_train[:-n_val]
print("Train:", x_train.shape, " Validation:", x_val.shape, " Test:", x_test.shape)

# 2. Mã hoá nhãn thành chuỗi token: PAD=0, SOS=1 (bắt đầu), EOS=2 (kết thúc), chữ số d -> d+3
PAD, SOS, EOS = 0, 1, 2
n_tokens = 13
seq_len = 6        # tối đa 5 chữ số + EOS

def encode_labels(labels):
    decoder_input = np.zeros((len(labels), seq_len), dtype='int32')
    decoder_target = np.zeros((len(labels), seq_len), dtype='int32')
    for i, s in enumerate(labels):
        tokens = [int(c) + 3 for c in s]
        decoder_input[i, :len(tokens) + 1] = [SOS] + tokens          # đầu vào decoder (teacher forcing)
        decoder_target[i, :len(tokens) + 1] = tokens + [EOS]         # đầu ra mong muốn
    return decoder_input, decoder_target

dec_in_train, dec_out_train = encode_labels(labels_train)
dec_in_val, dec_out_val = encode_labels(labels_val)
print("Nhãn:", labels_train[0], " decoder input:", dec_in_train[0], " decoder target:", dec_out_train[0])

# không tính loss ở các vị trí PAD
weight_train = (dec_out_train != PAD).astype('float32')
weight_val = (dec_out_val != PAD).astype('float32')

# 3. Encoder: CNN trích xuất đặc trưng ảnh
image_input = Input(shape=(64, 128, 3))
x = Rescaling(1.0 / 255)(image_input)
# Data augmentation (chỉ hoạt động khi train)
x = RandomTranslation(0.05, 0.05)(x)
x = RandomRotation(0.02)(x)
x = RandomContrast(0.3)(x)

x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)          # 32 x 64

x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)          # 16 x 32

x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)          # 8 x 16

x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)          # 4 x 8
x = Dropout(0.25)(x)

# Mỗi ô trong lưới 4 x 8 là một vector đặc trưng -> chuỗi 32 vector cho attention
encoder_outputs = Reshape((4 * 8, 256))(x)
image_vector = GlobalAveragePooling1D()(encoder_outputs)
state_h = Dense(256, activation='tanh')(image_vector)      # trạng thái khởi tạo cho decoder
state_c = Dense(256, activation='tanh')(image_vector)

# 4. Decoder: LSTM sinh lần lượt từng chữ số, có Attention lên các vùng ảnh
decoder_input = Input(shape=(seq_len,), dtype='int32')
y = Embedding(n_tokens, 64)(decoder_input)
decoder_outputs = LSTM(256, return_sequences=True)(y, initial_state=[state_h, state_c])
context, attention_scores = AdditiveAttention()([decoder_outputs, encoder_outputs], return_attention_scores=True)
y = Concatenate()([decoder_outputs, context])
y = Dense(256, activation='tanh')(y)
y = Dropout(0.3)(y)
outputs = Dense(n_tokens, activation='softmax')(y)

model = Model(inputs=[image_input, decoder_input], outputs=outputs)
model.summary()
attention_model = Model(inputs=[image_input, decoder_input], outputs=attention_scores)

# 5. Compile và train
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
H = model.fit([x_train, dec_in_train], dec_out_train, sample_weight=weight_train,
              validation_data=([x_val, dec_in_val], dec_out_val, weight_val),
              epochs=40, batch_size=128, verbose=2, callbacks=[early_stop, reduce_lr])

# 6. Dự đoán: sinh từng token (greedy decoding), bắt đầu từ SOS
def predict_sequences(images):
    dec = np.zeros((len(images), seq_len), dtype='int32')
    dec[:, 0] = SOS
    tokens = np.zeros((len(images), seq_len), dtype='int32')
    for t in range(seq_len):
        probs = model.predict([images, dec], batch_size=256, verbose=0)
        tokens[:, t] = probs[:, t].argmax(axis=-1)
        if t + 1 < seq_len:
            dec[:, t + 1] = tokens[:, t]
    results = []
    for row in tokens:
        s = ''
        for tok in row:
            if tok == EOS or tok < 3:
                break
            s += str(tok - 3)
        results.append(s)
    return np.array(results), dec

pred_test, dec_pred_test = predict_sequences(x_test)
for i in range(10):
    print("Thực tế:", labels_test[i], "  Dự đoán:", pred_test[i])

# 7. Đánh giá
def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]

def nse(y_true, y_pred):
    return 1 - sum((y_true - y_pred)**2) / sum((y_true - mean(y_true))**2)

def r2(y_true, y_pred):
    return corrcoef(y_true, y_pred)[0, 1]**2

def evaluate(y_true, y_pred):
    return {'R2': r2(y_true, y_pred), 'NSE': nse(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred), 'RMSE': sqrt(mean_squared_error(y_true, y_pred))}

sequence_accuracy = mean(pred_test == labels_test)
char_errors = sum(edit_distance(p, t) for p, t in zip(pred_test, labels_test))
cer = char_errors / sum(len(t) for t in labels_test)
length_accuracy = mean([len(p) == len(t) for p, t in zip(pred_test, labels_test)])

# Đổi chuỗi chữ số thành giá trị số để tính R2, NSE, MAE, RMSE
value_true = np.array([int(s) for s in labels_test], dtype='float64')
value_pred = np.array([int(s) if s != '' else 0 for s in pred_test], dtype='float64')
result_value = evaluate(value_true, value_pred)
result_log = evaluate(log10(1 + value_true), log10(1 + value_pred))

print('Sequence accuracy:', sequence_accuracy)
print('Character error rate:', cer, ' Length accuracy:', length_accuracy)
print('Numeric value metrics:', result_value)
print('log10(1+value) metrics:', result_log)

with open(out_dir + 'results.json', 'w') as f:
    json.dump({'n_samples': {'train': len(x_train), 'val': len(x_val), 'test': len(x_test)},
               'epochs_trained': len(H.history['loss']), 'params': model.count_params(),
               'sequence_accuracy': sequence_accuracy, 'character_error_rate': cer,
               'character_accuracy': 1 - cer, 'length_accuracy': length_accuracy,
               'value_metrics': result_value, 'log_value_metrics': result_log,
               'length_distribution_test': {str(k): int(sum(len(t) == k for t in labels_test)) for k in range(1, 6)}},
              f, indent=2, default=float)

# 8. Hình
plt.figure(figsize=(6, 3.5))
plt.plot(H.history['loss'], label='train')
plt.plot(H.history['val_loss'], label='validation')
plt.xlabel('epoch'); plt.ylabel('cross-entropy'); plt.title('Task 3 - learning curve'); plt.legend()
plt.tight_layout(); plt.savefig(out_dir + 'loss.png', dpi=150); plt.close()

plt.figure(figsize=(10, 5))
wrong = np.where(pred_test != labels_test)[0]
show = list(range(8)) + list(wrong[wrong >= 8][:4])     # 8 ảnh đầu + 4 ảnh dự đoán sai (không trùng)
for k, i in enumerate(show):
    plt.subplot(3, 4, k + 1)
    plt.imshow(x_test[i])
    plt.title('true ' + labels_test[i] + ' / pred ' + pred_test[i], fontsize=9,
              color='green' if pred_test[i] == labels_test[i] else 'red')
    plt.axis('off')
plt.tight_layout(); plt.savefig(out_dir + 'predictions.png', dpi=150); plt.close()

# Bản đồ attention: mỗi chữ số được sinh ra nhìn vào vùng nào của ảnh
i = [j for j in range(len(labels_test)) if len(labels_test[j]) >= 3 and pred_test[j] == labels_test[j]][0]
scores = attention_model.predict([x_test[i:i + 1], dec_pred_test[i:i + 1]], verbose=0)[0]
n_steps_show = len(pred_test[i]) + 1
plt.figure(figsize=(3 * n_steps_show, 2.2))
for t in range(n_steps_show):
    plt.subplot(1, n_steps_show, t + 1)
    plt.imshow(x_test[i])
    heat = np.kron(scores[t].reshape(4, 8), np.ones((16, 16)))
    plt.imshow(heat, alpha=0.5, cmap='jet')
    plt.title('step %d: %s' % (t + 1, pred_test[i][t] if t < len(pred_test[i]) else 'EOS'))
    plt.axis('off')
plt.tight_layout(); plt.savefig(out_dir + 'attention.png', dpi=150); plt.close()
