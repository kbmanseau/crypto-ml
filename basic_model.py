import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
from util import csv_to_dataset, history_points


# dataset

ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('ETHUSDT-1h-data.csv')

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)


# model architecture
bs = 1024
e = 500

lstm_input = tf.keras.layers.Input(shape=(history_points, 5), name='lstm_input')
x = tf.keras.layers.LSTM(50, name='lstm_0')(lstm_input)
x = tf.keras.layers.Dropout(0.2, name='lstm_dropout_0')(x)
x = tf.keras.layers.Dense(64, name='dense_0')(x)
x = tf.keras.layers.Activation('sigmoid', name='sigmoid_0')(x)
x = tf.keras.layers.Dense(1, name='dense_1')(x)
output = tf.keras.layers.Activation('linear', name='linear_output')(x)

model = tf.keras.models.Model(inputs=lstm_input, outputs=output)
adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=bs, epochs=e, shuffle=True, validation_split=0.1)


# evaluation

y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

from datetime import datetime
model.save(f'basic_model.h5')
