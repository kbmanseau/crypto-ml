import tensorflow as tf
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
from util import csv_to_dataset, history_points


# dataset

#File that will be used
csv_path = "ETHUSDT-1h-data.csv"

#ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('ETHUSDT-1h-data.csv')
ohlcv_histories_train, ohlcv_histories_test, technical_indicators_train, technical_indicators_test, unscaled_y_test, y_train, y_test, y_normaliser, tech_ind_train, tech_ind_test = csv_to_dataset(csv_path)

# model architecture
bs = 1024
e = 4000


# define two sets of inputs
lstm_input = tf.keras.layers.Input(shape=(history_points, 5), name='lstm_input')
dense_input = tf.keras.layers.Input(shape=(technical_indicators_train.shape[1],), name='tech_input')

# the first branch operates on the first input
x = tf.keras.layers.LSTM(50, name='lstm_0')(lstm_input)
x = tf.keras.layers.Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = tf.keras.models.Model(inputs=lstm_input, outputs=x)

# the second branch opreates on the second input
y = tf.keras.layers.Dense(20, name='tech_dense_0')(dense_input)
y = tf.keras.layers.Activation("relu", name='tech_relu_0')(y)
y = tf.keras.layers.Dropout(0.2, name='tech_dropout_0')(y)
technical_indicators_branch = tf.keras.models.Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = tf.keras.layers.concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

z = tf.keras.layers.Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = tf.keras.layers.Dense(1, activation="linear", name='dense_out')(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = tf.keras.models.Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=[ohlcv_histories_train, tech_ind_train], y=y_train, batch_size=bs, epochs=e, shuffle=True, validation_split=0.1)


# evaluation

y_test_predicted = model.predict([ohlcv_histories_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories_train, technical_indicators_train])
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

plt.savefig("test_data.png")
plt.show()

from datetime import datetime
model.save(f'technical_model.h5')
