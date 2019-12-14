import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

history_points = 168


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)

    #remove columns that are not needed
    #timestamp, close_time, quote_av, trades, tb_base_av, tb_quote_av, ignore
    data = data.drop('timestamp', axis=1)
    data = data.drop('close_time', axis=1)
    data = data.drop('quote_av', axis=1)
    data = data.drop('trades', axis=1)
    data = data.drop('tb_base_av', axis=1)
    data = data.drop('tb_quote_av', axis=1)
    data = data.drop('ignore', axis=1)

    #Split the data into train and test sets before normalization
    test_split = 0.9
    n = int(len(data) * test_split)

    d_train = data[:n]
    d_test = data[n:]
    #d_train, d_test = train_test_split(data, train_size=train_size, test_size=test_size, shuffle=False)

    #Scale the data. Test data is scaled separately but with the same scaling parameters as the test data
    data_normaliser = preprocessing.MinMaxScaler()
    train_normalised = data_normaliser.fit_transform(d_train)
    test_normalised = data_normaliser.transform(d_test)

    #Get the data ready for model consumption
    #using the last {history_points} open high low close volume data points, predict the next open value
    #ohlcv_histories_normalised is a three dimensional array of size (len(data_normalised), history points, 5(open, high, low, close, volume))
    #Each item in the list is an array of 50 days worth of ohlcv
    ohlcv_train = np.array([train_normalised[i:i + history_points].copy() for i in range(len(train_normalised) - history_points)])
    ohlcv_test = np.array([test_normalised[i:i + history_points].copy() for i in range(len(test_normalised) - history_points)])

    #What is being predicted, will be compared for error
    #ndov = next day open values
    ndov_train_normalised = np.array([train_normalised[:, 0][i + history_points].copy() for i in range(len(train_normalised) - history_points)])
    ndov_train_normalised = np.expand_dims(ndov_train_normalised, -1)

    ndov_test_normalised = np.array([test_normalised[:, 0][i + history_points].copy() for i in range(len(test_normalised) - history_points)])
    ndov_test_normalised = np.expand_dims(ndov_test_normalised, -1)

    #Unormalised data for plotting later
    ndov_train = np.array([d_train['open'][i + history_points].copy() for i in range(len(d_train) - history_points)])
    ndov_train = np.expand_dims(ndov_train, -1)

    ndov_test = np.array([d_test['open'][i + len(d_train) + history_points].copy() for i in range(len(d_test) - history_points)])
    ndov_test = np.expand_dims(ndov_test, -1)

    #Variables to scale the data back up later
    y_normaliser = preprocessing.MinMaxScaler()
    y_train = y_normaliser.fit_transform(ndov_train)
    y_test = y_normaliser.transform(ndov_test)

    #def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    #    sma = np.mean(values[:, 3])
    #    ema_values = [sma]
    #    k = 2 / (1 + time_period)
    #    for i in range(len(his) - time_period, len(his)):
    #        close = his[i][3]
    #        ema_values.append(close * k + ema_values[-1] * (1 - k))
    #    return ema_values[-1]

    #tis - technical indicators
    tis_train = []
    tis_test = []
    for his in ohlcv_train:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        #macd = calc_ema(his, 12) - calc_ema(his, 26)
        tis_train.append(np.array([sma]))
        # technical_indicators.append(np.array([sma,macd,]))
    for his in ohlcv_test:
        sma = np.mean(his[:, 3])
        tis_test.append(np.array([sma]))

    tis_train = np.array(tis_train)
    tis_test = np.array(tis_test)

    indicator_normaliser = preprocessing.MinMaxScaler()
    tis_normalised_train = indicator_normaliser.fit_transform(tis_train)
    tis_normalised_test = indicator_normaliser.transform(tis_test)

    #assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    #return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
    return ohlcv_train, \
        ohlcv_test, \
        tis_normalised_train, \
        tis_normalised_test, \
        ndov_test, \
        y_train, \
        y_test, \
        y_normaliser, \
        tis_normalised_train, \
        tis_normalised_test


#def multiple_csv_to_dataset(test_set_name):
#    import os
#    ohlcv_histories = 0
#    technical_indicators = 0
#    next_day_open_values = 0
#    for csv_file_path in list(filter(lambda x: x.endswith('daily.csv'), os.listdir('./'))):
#        if not csv_file_path == test_set_name:
#            print(csv_file_path)
#            if type(ohlcv_histories) == int:
#                ohlcv_histories, technical_indicators, next_day_open_values, _, _ = csv_to_dataset(csv_file_path)
#            else:
#                a, b, c, _, _ = csv_to_dataset(csv_file_path)
#                ohlcv_histories = np.concatenate((ohlcv_histories, a), 0)
#                technical_indicators = np.concatenate((technical_indicators, b), 0)
#                next_day_open_values = np.concatenate((next_day_open_values, c), 0)#
#
#    ohlcv_train = ohlcv_histories
#    tech_ind_train = technical_indicators
#    y_train = next_day_open_values
#
#    ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser = csv_to_dataset(test_set_name)
#
#    return ohlcv_train, tech_ind_train, y_train, ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser
