import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(1)
tf.random.set_seed(1)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Data Downloading

df = df_raw[['timestamp', 'Momentary energy consumption']]

# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
df['Momentary energy consumption'] = scaler.fit_transform(df['Momentary energy consumption'].values.reshape(-1, 1))


sequence_length = 24
data = []
labels = []
for i in range(len(df) - sequence_length):
    data.append(df['Momentary energy consumption'].values[i:i + sequence_length])
    labels.append(df['Momentary energy consumption'].values[i + sequence_length])

data = np.array(data)
labels = np.array(labels)

#Divide data for 3 local virtual devices
X_train_1, y_train_1 = data[0:4000], labels[0:4000]
X_train_2, y_train_2 = data[4000:8000], labels[4000:8000]
X_train_3, y_train_3 = data[8000:12000], labels[8000:12000]

X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], X_train_1.shape[1], 1))
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], 1))
X_train_3 = np.reshape(X_train_3, (X_train_3.shape[0], X_train_3.shape[1], 1))

#  MSE calculation
def calculate_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# LSTM model reation
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=64, input_shape=input_shape),
        Dense(units=1)
    ])
    return model

# Training
local_model_1 = create_lstm_model((sequence_length, 1))
local_model_1.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
local_model_1.fit(X_train_1, y_train_1, epochs=30, verbose=0)
weight1 = local_model_1.get_weights()

local_model_2 = create_lstm_model((sequence_length, 1))
local_model_2.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
local_model_2.fit(X_train_2, y_train_2, epochs=30, verbose=0)
weight2 = local_model_2.get_weights()

local_model_3 = create_lstm_model((sequence_length, 1))
local_model_3.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
local_model_3.fit(X_train_3, y_train_3, epochs=30, verbose=0)
weight3 = local_model_3.get_weights()

# Test data frame
tdf = df[12000:12500]


test_data = []
test_labels = []
for i in range(len(tdf) - sequence_length):
    test_data.append(tdf['Momentary energy consumption'].values[i:i + sequence_length])
    test_labels.append(tdf['Momentary energy consumption'].values[i + sequence_length])

test_data = np.array(test_data)
test_labels = np.array(test_labels)

X_test = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
y_test = test_labels

loss1 = calculate_mse(y_test, local_model_1.predict(X_test))
loss2 = calculate_mse(y_test, local_model_2.predict(X_test))
loss3 = calculate_mse(y_test, local_model_3.predict(X_test))


initial_weights = [weight1, weight2, weight3]
losses = [loss1, loss2, loss3]

# GLobal model formation
def aggregate_weights(client_weights, losses):
    c = sum(losses)
    w1w = (1 - (losses[0] / c)) / 1.5
    w2w = (1 - (losses[1] / c)) / 1.5
    w3w = (1 - (losses[2] / c)) #/ 2

    print(f"w1w: {w1w}, w2w: {w2w}, w3w: {w3w}")

    global_weights = [w1w * w + w2w * client_weights[1][i] + w3w * client_weights[2][i] 
                      for i, w in enumerate(client_weights[0])]
    
    return global_weights

global_weights = aggregate_weights(initial_weights, losses)


global_model = create_lstm_model((sequence_length, 1))
global_model.set_weights(global_weights)
global_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')


predictions = global_model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Momentary energy consumption')
plt.legend()
plt.show()
