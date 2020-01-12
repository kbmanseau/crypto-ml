import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy as np

history_points = 14
scaler_x_filename = "x_scaler.save"
scaler_td_filename = "td_scaler.save"
scaler_ti_filename = "ti_scaler.save"

def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)

    #remove columns that are not needed
    #timestamp, close_time, quote_av, trades, tb_base_av, tb_quote_av, ignore
    data = data.drop('counter', axis=1)
    data = data.drop('timestamp', axis=1)
    data = data.drop('close_time', axis=1)
    data = data.drop('quote_av', axis=1)
    data = data.drop('trades', axis=1)
    data = data.drop('tb_base_av', axis=1)
    data = data.drop('tb_quote_av', axis=1)
    data = data.drop('ignore', axis=1)

    td_data = data
    data = data.drop('td', axis=1)

    #Split the data into train and test sets before normalization
    test_split = 0.9
    n = int(len(data) * test_split)

    d_train = data[:n]
    td_train = td_data[:n]
    d_test = data[n:]
    td_test = td_data[n:]

    #Scale the data. Test data is scaled separately but with the same scaling parameters as the test data
    data_normaliser = preprocessing.MinMaxScaler()
    train_normalised = data_normaliser.fit_transform(d_train)
    test_normalised = data_normaliser.transform(d_test)

    #Save the scaler to scale new data for predictions
    joblib.dump(data_normaliser, scaler_x_filename)

    #Get the data ready for model consumption
    #using the last {history_points} open high low close volume data points, predict the next open value
    #ohlcv_histories_normalised is a three dimensional array of size (len(data_normalised), history points, 5(open, high, low, close, volume))
    #Each item in the list is an array of 50 days worth of ohlcv
    ohlcv_train = np.array([train_normalised[i:i + history_points].copy() for i in range(len(train_normalised) - history_points)])
    ohlcv_test = np.array([test_normalised[i:i + history_points].copy() for i in range(len(test_normalised) - history_points)])

    #Values model is trying to predict
    y_train = np.array(td_train['td'])
    y_train = np.expand_dims(y_train, -1)
    #Strip of initial history_points amount of data points since the model needs atleast history_points history
    y_train = y_train[history_points:]


    y_test = np.array(td_test['td'])
    y_test = np.expand_dims(y_test, -1)
    #Strip of initial history_points amount of data points since the model needs atleast history_points history
    y_test = y_test[history_points:]

    #Scale the data. Test data is scaled separately but with the same scaling parameters as the test data
    td_normaliser = preprocessing.MinMaxScaler()
    y_train = td_normaliser.fit_transform(y_train)
    y_test = td_normaliser.transform(y_test)

    #Save the scaler to scale new data for predictions
    joblib.dump(td_normaliser, scaler_td_filename)

    #tis - technical indicators
    tis_train = []
    tis_test = []
    for his in ohlcv_train:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        tis_train.append(np.array([sma]))
    for his in ohlcv_test:
        sma = np.mean(his[:, 3])
        tis_test.append(np.array([sma]))

    tis_train = np.array(tis_train)
    tis_test = np.array(tis_test)

    indicator_normaliser = preprocessing.MinMaxScaler()
    tis_normalised_train = indicator_normaliser.fit_transform(tis_train)
    tis_normalised_test = indicator_normaliser.transform(tis_test)

    #Save the scaler of the technical indicators
    joblib.dump(indicator_normaliser, scaler_ti_filename)

    #assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    #return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

    return ohlcv_train, \
        ohlcv_test, \
        y_train, \
        y_test, \
        tis_normalised_train, \
        tis_normalised_test
