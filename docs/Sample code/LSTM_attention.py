# lstm for time series forecasting
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, Attention, GlobalAveragePooling1D, Input, Dropout
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model

# split a univariate sequence into samples
def split_sequence(sequence, n_steps, n_pred, pred_col):
    X, y = list(), list()
    for i in range(len(sequence) - (n_steps + n_pred) +1):
        # gather input and output parts of the pattern
        seq_x = sequence[i: i+n_steps, :]
        seq_y = sequence[i+n_steps+n_pred -1, pred_col]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

# load the dataset
filename = './Data/LSTM-Multivariate_pollution.csv'
all_attributes = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data = read_csv(filename, index_col=False, usecols=all_attributes, encoding='utf-8')
print(data)
#Chuyển dữ liệu chuỗi sang số nguyên
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['wnd_dir'] = le.fit_transform(data['wnd_dir'])
print(data)
# retrieve the values
values = data.values.astype('float32')
print(values)
# specify the window size
n_steps = 3
n_pred = 2
pred_col=0
# split into samples
X, y = split_sequence(values, n_steps, n_pred, pred_col)
print("X0:", X[0])
print("y0:", y[0])
print("X-1:", X[-1])
print("y-1:", y[-1])
print("y=",y)
# reshape into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
# split into train/test
n_test = 15000
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=false)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_val=X_test
y_val=y_test
# define model
# Input
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# SimpleRNN
rnn_output = SimpleRNN(100, activation='relu', kernel_initializer='he_normal', return_sequences=True)(inputs)

# Attention: Query = Value = RNN output
attention_output = Attention()([rnn_output, rnn_output])

# Tổng hợp thông tin theo thời gian
x = GlobalAveragePooling1D()(attention_output)

# Dense layers
x = Dense(20, activation='relu', kernel_initializer='he_normal')(x)
outputs = Dense(1)(x)

# Build model
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# fit the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_val, y_val))
# make a prediction
y_pred = model.predict(X_test)
#evaluate the model
print("y_test=", y_test)
print("y_pred=", y_pred)
y_pred=y_pred.reshape(y_pred.shape[0])
print(y_pred)
print('R2: ', r2_score(y_test, y_pred))