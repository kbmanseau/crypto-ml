import pandas as pd
import os.path
import math
import argparse
import json
from binance.client import Client
from datetime import datetime
from dateutil import parser as par

def time_differencing(td_data_df):
    for i in td_data_df.index:
        if i < len(td_data_df)-1:
            td_data_df.loc[i,"td"] = float(td_data_df.loc[i+1,"open"]) - float(td_data_df.loc[i, "open"])
        else:
            td_data_df.loc[i,'td'] = 0

    return td_data_df

def get_all_binance(symbol, kline_size, data_type, save = True):
    #https://medium.com/swlh/retrieving-full-historical-data-for-every-cryptocurrency-on-binance-bitmex-using-the-python-apis-27b47fd8137f

    #Only download new data instead of downloading the entire set every runtime
    def minutes_of_new_data(symbol, kline_size, data, client, source):
        if len(data) > 0:
            old = par.parse(data["timestamp"].iloc[-1])
        elif source == "binance":
            old = datetime.strptime('1 Jan 2017', '%d %b %Y')
        if source == "binance":
            new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
        return old, new 
    
    ### API
    binance_creds = json.load(open('api.json', 'r'))
    binance_api_key = binance_creds['key']
    binance_api_secret = binance_creds['secret']
    binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

    #Variables
    binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
    if data_type!="td":
        filename = '%s-%s-data.csv' % (symbol, kline_size)
    else:
        filename = '%s-%s-data-td.csv' % (symbol, kline_size)

    #If file exists, read it, otherwise create new object
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame()
    
    #Perform checks for only downloading new data
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, binance_client, source = "binance")
    if data_type == "reg":
        delta_min = (newest_point - oldest_point).total_seconds()/60
        available_data = math.ceil(delta_min/binsizes[kline_size])
        if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'):
            print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
        else:
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    elif data_type == "td":
        delta_day = (newest_point - oldest_point).total_seconds()/86400
        available_data = math.ceil(delta_day/binsizes[kline_size])
        if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'):
            print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
        else:
            print('Downloading %d days of new data available for %s, i.e. %d instances of %s data.' % (delta_day, symbol, available_data, kline_size))
    
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    if len(data_df) > 0:
        if(oldest_point != newest_point):
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df, sort=False)
    else:
        data_df = data

    if data_type == "td":
        #Create column allowing for referincing index by an int
        x = 0
        for i in data_df.index:
            data_df.loc[i, 'counter'] = x
            x = x+1

        data_df.set_index('counter', inplace=True)
    else:
        data_df.set_index('timestamp', inplace=True)

    if save:
        if data_type == "td": #Saved format will be time differenced
            data_df = time_differencing(data_df)
        data_df.to_csv(filename)
        
    print('All caught up..!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('symbol', type=str, help="the crypto pair on binance you want to download Example 'ETHUSDT'")
    parser.add_argument('kline_size', type=str, choices=[
                        '1h', '1d'], help="the time period you want to download the stock history for")
    parser.add_argument('data_type', type=str, choices=['reg','td'], help="Either regular price data or time-differenced dataset")
    
    namespace = parser.parse_args()
    get_all_binance(**vars(namespace))
