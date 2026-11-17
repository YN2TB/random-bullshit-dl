# Chuẩn bị dữ liệu SVHN (format 1 - full numbers) cho Task 3
# Nguồn: http://ufldl.stanford.edu/housenumbers/  (tải train.tar.gz, test.tar.gz về 01_data/raw/svhn/,
# script tự giải nén nếu chưa có thư mục train/, test/)
# Chạy từ thư mục gốc project:  .venv/Scripts/python 01_data/prepare_svhn.py
import os
import tarfile
import h5py
import numpy as np
from PIL import Image

raw_dir = '01_data/raw/svhn/'
out_dir = '01_data/processed/'
os.makedirs(out_dir, exist_ok=True)
img_h, img_w = 64, 128     # kích thước ảnh sau khi cắt và resize
max_len = 5                # chỉ giữ số nhà có tối đa 5 chữ số

# Giải nén train.tar.gz, test.tar.gz (chỉ làm lần đầu)
for split in ['train', 'test']:
    if not os.path.exists(raw_dir + split + '/digitStruct.mat'):
        print('Giải nén', split + '.tar.gz ...')
        with tarfile.open(raw_dir + split + '.tar.gz') as tar:
            tar.extractall(raw_dir, filter='data')

# Đọc nhãn và bounding box của từng chữ số trong file digitStruct.mat (định dạng HDF5)
def read_digit_struct(mat_file):
    f = h5py.File(mat_file, 'r')
    names = f['digitStruct/name']
    bboxes = f['digitStruct/bbox']
    records = []
    for i in range(len(names)):
        name = ''.join(chr(c[0]) for c in f[names[i][0]][()])
        box = f[bboxes[i][0]]
        values = {}
        for key in ['label', 'left', 'top', 'width', 'height']:
            item = box[key]
            if item.shape[0] == 1:   # chỉ có 1 chữ số: giá trị lưu trực tiếp
                values[key] = [float(item[0][0])]
            else:                    # nhiều chữ số: mỗi phần tử là một reference
                values[key] = [float(f[item[j][0]][()][0][0]) for j in range(item.shape[0])]
        records.append((name, values))
        if i % 5000 == 0:
            print(mat_file, i, '/', len(names))
    return records

# Cắt vùng chứa toàn bộ dãy số (nới rộng 30%), resize về img_h x img_w
def load_split(split):
    records = read_digit_struct(raw_dir + split + '/digitStruct.mat')
    X, labels, n_drop = [], [], 0
    for name, v in records:
        digits = [0 if d == 10 else int(d) for d in v['label']]   # nhãn 10 nghĩa là chữ số 0
        if len(digits) > max_len:
            n_drop += 1
            continue
        img = Image.open(raw_dir + split + '/' + name).convert('RGB')
        left = min(v['left'])
        top = min(v['top'])
        right = max(l + w for l, w in zip(v['left'], v['width']))
        bottom = max(t + h for t, h in zip(v['top'], v['height']))
        dw, dh = 0.15 * (right - left), 0.15 * (bottom - top)
        box = (max(0, left - dw), max(0, top - dh), min(img.width, right + dw), min(img.height, bottom + dh))
        img = img.crop(box).resize((img_w, img_h), Image.BILINEAR)
        X.append(np.asarray(img, dtype='uint8'))
        labels.append(''.join(str(d) for d in digits))
    print(split, ': giữ', len(X), 'ảnh, bỏ', n_drop, 'ảnh có hơn', max_len, 'chữ số')
    return np.stack(X), np.array(labels), n_drop

X_train, labels_train, drop_train = load_split('train')
X_test, labels_test, drop_test = load_split('test')
import json
json.dump({'train_raw': len(X_train) + drop_train, 'test_raw': len(X_test) + drop_test,
           'dropped_train': drop_train, 'dropped_test': drop_test, 'image_size': [img_h, img_w, 3]},
          open(out_dir + 'svhn_info.json', 'w'), indent=2)
print('X_train:', X_train.shape, ' X_test:', X_test.shape)
print('Ví dụ nhãn:', labels_train[:10])

np.savez_compressed(out_dir + 'svhn_64x128.npz', X_train=X_train, labels_train=labels_train,
                    X_test=X_test, labels_test=labels_test)

# Lưu vài ảnh mẫu để kiểm tra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_train[i])
    plt.title(labels_train[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig(out_dir + 'svhn_samples.png', dpi=120)
