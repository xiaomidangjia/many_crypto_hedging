import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
#from tqdm import tqdm
from datetime import datetime,timedelta
from send_email import email_sender
#=====定义函数====
from HTMLTable import HTMLTable
'''
生成html表格
传入一个dataframe, 设置一个标题， 返回一个html格式的表格
'''
def create_html_table(df, title):
    table = HTMLTable(caption=title)

    # 表头行
    table.append_header_rows((tuple(df.columns),))

    # 数据行
    for i in range(len(df.index)):
        table.append_data_rows((
            tuple(df.iloc[df.index[i],]),
        ))

    # 标题样式
    table.caption.set_style({
        'font-size': '15px',
    })

    # 表格样式，即<table>标签样式
    table.set_style({
        'border-collapse': 'collapse',
        'word-break': 'keep-all',
        'white-space': 'nowrap',
        'font-size': '14px',
    })

    # 统一设置所有单元格样式，<td>或<th>
    table.set_cell_style({
        'border-color': '#000',
        'border-width': '1px',
        'border-style': 'solid',
        'padding': '5px',
        'text-align': 'center',
    })

    # 表头样式
    table.set_header_row_style({
        'color': '#fff',
        'background-color': '#48a6fb',
        'font-size': '15px',
    })

    # 覆盖表头单元格字体样式
    table.set_header_cell_style({
        'padding': '15px',
    })

    # 调小次表头字体大小
    table[0].set_cell_style({
        'padding': '8px',
        'font-size': '15px',
    })

    html_table = table.to_html()
    return html_table

# ======= 正式开始执行
date_now = str(datetime.utcnow())[0:10]

# 获取数据

import importlib
import sys
import os
import urllib
import requests
import base64
import json

import time
import pandas as pd
import numpy as np
import random
import hmac
import ccxt
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')
# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df
API_URL = 'https://api.bitget.com'
API_SECRET_KEY = 'ca8d708b774782ce0fd09c78ba5c19e1e421d5fd2a78964359e6eb306cf15c67'
API_KEY = 'bg_42d96db83714abb3757250cef9ba7752'
PASSPHRASE = 'HBLww130130130'
margein_coin = 'USDT'
futures_type = 'USDT-FUTURES'
contract_num = 5

def get_timestamp():
    return int(time.time() * 1000)
def sign(message, secret_key):
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d)
def pre_hash(timestamp, method, request_path, body):
    return str(timestamp) + str.upper(method) + request_path + body
def parse_params_to_str(params):
    url = '?'
    for key, value in params.items():
        url = url + str(key) + '=' + str(value) + '&'
    return url[0:-1]
def get_header(api_key, sign, timestamp, passphrase):
    header = dict()
    header['Content-Type'] = 'application/json'
    header['ACCESS-KEY'] = api_key
    header['ACCESS-SIGN'] = sign
    header['ACCESS-TIMESTAMP'] = str(timestamp)
    header['ACCESS-PASSPHRASE'] = passphrase
    # header[LOCALE] = 'zh-CN'
    return header

def truncate(number, decimals):
    factor = 10.0 ** decimals
    return int(number * factor) / factor

def write_txt(content):
    with open(f"/home/liweiwei/duichong/test/process_result.txt", "a") as file:
        file.write(content)

def get_price(symbol):
    w2 = 0
    g2 = 0 
    while w2 == 0:
        try:
            timestamp = get_timestamp()
            response = None
            request_path = "/api/v2/mix/market/ticker"
            url = API_URL + request_path
            params = {"symbol":symbol,"productType":futures_type}
            request_path = request_path + parse_params_to_str(params)
            url = API_URL + request_path
            body = ""
            sign_cang = sign(pre_hash(timestamp, "GET", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_cang, timestamp, PASSPHRASE)
            response = requests.get(url, headers=header)
            ticker = json.loads(response.text)
            price_d = float(ticker['data'][0]['lastPr'])
            if price_d > 0:
                w2 = 1
            else:
                w2 = 0
            g2 += 1

        except:
            time.sleep(0.2)
            g2 += 1
            continue
    return price_d

def open_state(crypto_usdt,order_usdt,side,tradeSide):
    logo_b = 0
    while logo_b == 0:
        try:
            timestamp = get_timestamp()
            response = None
            clientoid = "bitget%s"%(str(int(datetime.now().timestamp())))
            #print('clientoid'+clientoid)
            request_path = "/api/v2/mix/order/place-order"
            url = API_URL + request_path
            params = {"symbol":crypto_usdt,"productType":futures_type,"marginCoin": margein_coin, "marginMode":"crossed","side":side,"tradeSide":tradeSide,"size":str(order_usdt),"orderType":"market","clientOid":clientoid}
            #print(params)
            body = json.dumps(params)
            sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
            response = requests.post(url, data=body, headers=header)
            buy_res_base = json.loads(response.text)
            #print("响应内容 (文本)---1:", buy_res_base)
            buy_id_base = int(buy_res_base['data']['orderId'])
            if buy_id_base  > 10:
                logo_b = 1
            else:
                logo_b = 0
        except:
            time.sleep(0.2)
            continue
    return buy_id_base

def check_order(crypto_usdt,id_num):
    logo_s = 0
    while logo_s == 0:
        try:
            timestamp = get_timestamp()
            response = None
            request_path_mix = "/api/v2/mix/order/detail"
            params_mix = {"symbol":crypto_usdt,"productType":futures_type,"orderId":str(id_num)}
            request_path_mix = request_path_mix + parse_params_to_str(params_mix)
            url = API_URL + request_path_mix
            body = ""
            sign_mix = sign(pre_hash(timestamp, "GET", request_path_mix,str(body)), API_SECRET_KEY)
            header_mix = get_header(API_KEY, sign_mix, timestamp, PASSPHRASE)
            response_mix = requests.get(url, headers=header_mix)

            response_1 = json.loads(response_mix.text)

            base_price = float(response_1['data']['priceAvg'])             
            base_num = float(response_1['data']['baseVolume'])
            if base_price >0 and base_num > 0:
                logo_s = 1
            else:
                logo_s = 0
        except:
            time.sleep(0.2)
            continue
    return base_price,base_num

def get_bitget_klines(symbol,endTime,granularity):
    timestamp = get_timestamp()
    response = None
    request_path_mix = "/api/v2/mix/market/candles"
    params_mix = {"symbol":symbol,"granularity":granularity,"productType":"USDT-FUTURES","endTime": endTime,"limit": 200}
    request_path_mix = request_path_mix + parse_params_to_str(params_mix)
    url = API_URL + request_path_mix
    body = ""
    sign_mix = sign(pre_hash(timestamp, "GET", request_path_mix,str(body)), API_SECRET_KEY)
    header_mix = get_header(API_KEY, sign_mix, timestamp, PASSPHRASE)
    response_mix = requests.get(url, headers=header_mix)
    response_1 = json.loads(response_mix.text)
    return response_1["data"]

def fetch_last_month_klines(symbol, granularity_value,number):
    """
    获取最近一个月的所有15分钟K线数据
    """
    klines = pd.DataFrame()
    # 计算一个月前的时间戳（毫秒）
    one_month_ago = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
    end_time = one_month_ago
    signal = 0
    while True:
        data = get_bitget_klines(symbol,endTime=end_time,granularity=granularity_value)
        res = pd.DataFrame()
        for i in range(len(data)):
            ins = data[i]
            date_time = ins[0]
            open_price = ins[1]
            high_price = ins[2]
            low_price = ins[3]
            close_price = ins[4]
            btc_volumn = ins[5]
            usdt_volumn = ins[6]
            # 秒级时间戳
            timestamp_seconds = int(date_time)/1000
            # 转换为正常时间
            normal_time = datetime.fromtimestamp(timestamp_seconds)
            # 格式化为字符串
            formatted_time = normal_time.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame({'date_time':date_time,'formatted_time':formatted_time,'open_price':open_price,'high_price':high_price,'low_price':low_price,'close_price':close_price,'btc_volumn':btc_volumn,'usdt_volumn':usdt_volumn},index=[0])
            res = pd.concat([res,df])
        #print(res)
        klines = pd.concat([klines,res])
        #print(klines)
        res = res.sort_values(by='date_time')
        res = res.reset_index(drop=True)
        # 更新下一个请求的开始时间为最后一条数据的时间戳
        last_time = int(res['date_time'][len(res)-1])
        #print(res['formatted_time'][0])
        #print(res['formatted_time'][len(res)-1])
        end_time = last_time + number * 1000 * 200  # 加上5分钟
        #print(end_time)

        # 如果获取的数据覆盖到当前时间，则停止循环
        
        if end_time <= int(time.time() * 1000) and signal ==0:
            continue
        elif end_time > int(time.time() * 1000) and signal ==0:
            signal = 1
            continue
        else:
            break

        # 避免频繁请求API，添加适当的延时
        time.sleep(1)

    return klines

coin_list = ['btc','sol','xrp','doge','eth']
for c_ele in coin_list:
    symbol = c_ele.upper() + 'USDT'
    data_1m_name = c_ele + '_1m_data.csv'
    data_15m_name = c_ele + '_15m_data.csv'
    data_1m = fetch_last_month_klines(symbol,granularity_value='1m',number=60)
    data_15m = fetch_last_month_klines(symbol,granularity_value='15m',number=900)
    data_1m.to_csv(data_1m_name)
    data_15m.to_csv(data_15m_name)


# ================================================================================== 第一个模型 =========================================================


btc_data = pd.read_csv('btc_15m_data.csv')
sol_data = pd.read_csv('sol_15m_data.csv')
eth_data = pd.read_csv('eth_15m_data.csv')
xrp_data = pd.read_csv('xrp_15m_data.csv')
doge_data = pd.read_csv('doge_15m_data.csv')

btc_data['date_time'] = btc_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
eth_data['date_time'] = eth_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
sol_data['date_time'] = sol_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
xrp_data['date_time'] = xrp_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
doge_data['date_time'] = doge_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))

btc_data = btc_data.drop('formatted_time', axis=1)
eth_data = eth_data.drop('formatted_time', axis=1)
sol_data = sol_data.drop('formatted_time', axis=1)
xrp_data = xrp_data.drop('formatted_time', axis=1)
doge_data = doge_data.drop('formatted_time', axis=1)

btc_data['date'] = btc_data['date_time'].apply(lambda x:x.date())
btc_data['hour'] = btc_data['date_time'].apply(lambda x:x.hour)
btc_data['minutes'] = btc_data['date_time'].apply(lambda x:x.minute)

eth_data['date'] = eth_data['date_time'].apply(lambda x:x.date())
eth_data['hour'] = eth_data['date_time'].apply(lambda x:x.hour)
eth_data['minutes'] = eth_data['date_time'].apply(lambda x:x.minute)

sol_data['date'] = sol_data['date_time'].apply(lambda x:x.date())
sol_data['hour'] = sol_data['date_time'].apply(lambda x:x.hour)
sol_data['minutes'] = sol_data['date_time'].apply(lambda x:x.minute)

xrp_data['date'] = xrp_data['date_time'].apply(lambda x:x.date())
xrp_data['hour'] = xrp_data['date_time'].apply(lambda x:x.hour)
xrp_data['minutes'] = xrp_data['date_time'].apply(lambda x:x.minute)

doge_data['date'] = doge_data['date_time'].apply(lambda x:x.date())
doge_data['hour'] = doge_data['date_time'].apply(lambda x:x.hour)
doge_data['minutes'] = doge_data['date_time'].apply(lambda x:x.minute)

sub_btc_data = btc_data[['date_time','close_price']]
sub_sol_data = sol_data[['date_time','close_price']]
sub_eth_data = eth_data[['date_time','close_price']]
sub_xrp_data = xrp_data[['date_time','close_price']]
sub_doge_data = doge_data[['date_time','close_price']]

sub_btc_data.rename(columns={'close_price':'btc_price'},inplace=True)
sub_sol_data.rename(columns={'close_price':'sol_price'},inplace=True)
sub_eth_data.rename(columns={'close_price':'eth_price'},inplace=True)
sub_xrp_data.rename(columns={'close_price':'xrp_price'},inplace=True)
sub_doge_data.rename(columns={'close_price':'doge_price'},inplace=True)

# 计算平均真实波动性
import pandas_ta as ta
def calculate_cci(data, n=20):
    """
    计算商品通道指数（CCI）。

    参数：
    data : pd.DataFrame
        包含 'high'、'low' 和 'close' 列的 DataFrame。
    n : int
        计算 CCI 的周期，默认值为 20。

    返回：
    pd.Series
        计算得到的 CCI 值。
    """
    # 计算典型价格（TP）
    tp = (data['high_price'] + data['low_price'] + data['close_price']) / 3

    # 计算 TP 的 N 日简单移动平均（SMA）
    sma_tp = tp.rolling(window=n, min_periods=1).mean()

    # 计算平均绝对偏差（Mean Deviation，MD）
    md = tp.rolling(window=n, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

    # 计算 CCI
    cci = (tp - sma_tp) / (0.015 * md)

    return cci

# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

# 计算价格变化率（Rate of Change）
def calculate_vol(df, column='usdt_volumn', window=14):
    df['VOL'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df

import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')
symbol_list = ['btc','eth','xrp','doge','sol']
art_df = pd.DataFrame()
roc_df = pd.DataFrame()
rsi_df = pd.DataFrame()
cci_df = pd.DataFrame()
for pair in itertools.combinations(symbol_list, 2):
    coin_1 = pair[0]
    coin_2 = pair[1]
    print(f"对比：{pair[0]} 和 {pair[1]}")
    if coin_1 == 'btc':
        coin1_data = sub_btc_data
        raw_coin1_data = btc_data
    elif coin_1 == 'eth':
        coin1_data = sub_eth_data
        raw_coin1_data = eth_data
    elif coin_1 == 'xrp':
        coin1_data = sub_xrp_data
        raw_coin1_data = xrp_data
    elif coin_1 == 'sol':
        coin1_data = sub_sol_data
        raw_coin1_data = sol_data
    elif coin_1 == 'doge':
        coin1_data = sub_doge_data
        raw_coin1_data = doge_data
    else:
        coin1_data = None
        raw_coin1_data = None
    if coin_2 == 'btc':
        coin2_data = sub_btc_data
        raw_coin2_data = btc_data
    elif coin_2 == 'eth':
        coin2_data = sub_eth_data
        raw_coin2_data = eth_data
    elif coin_2 == 'xrp':
        coin2_data = sub_xrp_data
        raw_coin2_data = xrp_data
    elif coin_2 == 'sol':
        coin2_data = sub_sol_data
        raw_coin2_data = sol_data
    elif coin_2 == 'doge':
        coin2_data = sub_doge_data
        raw_coin2_data = doge_data
    else:
        coin2_data = None
        raw_coin2_data = None 
    coin1_name = coin_1 + '_price'
    coin2_name = coin_2 + '_price'
    new_data = coin1_data.merge(coin2_data,how='left',on=['date_time'])
    new_data = new_data.dropna()
    new_data['date'] = new_data['date_time'].apply(lambda x:x.date())
    new_data['hour'] = new_data['date_time'].apply(lambda x:x.hour)
    new_data['minutes'] = new_data['date_time'].apply(lambda x:x.minute)
    date_period = list(sorted(set(new_data['date'])))
    length = len(date_period)
    #print('date_period')
    #print(date_period)
    i = 0
    while i < length-1:
        date_interval = date_period[i:i+2]
        #print('date_interval')
        #print(date_interval)
        sub_raw_coin1_data = raw_coin1_data[raw_coin1_data.date==date_interval[0]]
        sub_raw_coin2_data = raw_coin2_data[raw_coin2_data.date==date_interval[0]]
        sub_new_data = new_data[new_data.date==date_interval[1]]
        

        # 计算coin1的 atr
        # 计算coin1的 roc
        coin1_roc = calculate_roc(sub_raw_coin1_data)
        coin1_roc = coin1_roc.dropna()
        coin1_roc_value = np.mean(coin1_roc['ROC'])
        
        coin1_rsi = calculate_rsi(sub_raw_coin1_data)
        coin1_rsi = coin1_rsi.dropna()
        coin1_rsi_value = np.mean(coin1_rsi['RSI'])
        
        coin1_vol = calculate_vol(sub_raw_coin1_data)
        coin1_vol = coin1_vol.dropna()
        coin1_vol_value = np.mean(coin1_vol['VOL'])  
        
        
        coin1_cci = calculate_cci(sub_raw_coin1_data, n=20)
        coin1_cci = coin1_cci.dropna()
        coin1_cci_value = np.mean(coin1_cci)
        
        
        # 计算coin2的 atr
        # 计算coin2的 roc

        coin2_roc = calculate_roc(sub_raw_coin2_data)
        coin2_roc = coin2_roc.dropna()
        coin2_roc_value = np.mean(coin2_roc['ROC'])
        
        coin2_rsi = calculate_rsi(sub_raw_coin2_data)
        coin2_rsi = coin2_rsi.dropna()
        coin2_rsi_value = np.mean(coin2_rsi['RSI'])

        coin2_vol = calculate_vol(sub_raw_coin2_data)
        coin2_vol = coin2_vol.dropna()
        coin2_vol_value = np.mean(coin2_vol['VOL'])  
        
        coin2_cci = calculate_cci(sub_raw_coin2_data, n=20)
        coin2_cci = coin2_cci.dropna()
        coin2_cci_value = np.mean(coin2_cci)
        
        sub_new_data = sub_new_data.sort_values(by='date_time')
        coin1_name_change = coin1_name + '_change_pct'
        coin2_name_change = coin2_name + '_change_pct'

        # 获取第一行的值
        coin1_first_value = sub_new_data[coin1_name].iloc[0]
        sub_new_data[coin1_name_change] = (sub_new_data[coin1_name] - coin1_first_value) / coin1_first_value
        coin2_first_value = sub_new_data[coin2_name].iloc[0]
        sub_new_data[coin2_name_change] = (sub_new_data[coin2_name] - coin2_first_value) / coin2_first_value

        sub_new_data['price_change'] = sub_new_data[coin1_name_change] - sub_new_data[coin2_name_change]
        sub_new_data = sub_new_data.reset_index(drop=True)
        #print('sub_new_data')
        #print(sub_new_data)
        last_coin1_price = sub_new_data[coin1_name_change][len(sub_new_data)-1]
        last_coin2_price = sub_new_data[coin2_name_change][len(sub_new_data)-1]
        price_max = np.max(sub_new_data['price_change'])
        price_min = np.min(sub_new_data['price_change'])
        #print('price_max,price_min')
        #print(price_max,price_min)
        if price_max > 0.15 or price_min < -0.15:
            w = 0
            for j in range(len(sub_new_data)):
                price = sub_new_data['price_change'][j]
                if price > 0.15 :
                    # coin1为多,coin2为空
                    #ins_1 = pd.DataFrame({'long_coin':coin1_atr_value,'short_coin':coin2_atr_value,'coin1_name':coin1_name,'coin2_name':coin2_name},index=[0])
                    ins_2 = pd.DataFrame({'roc_long_coin':coin1_roc_value,'roc_short_coin':coin2_roc_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                    ins_3 = pd.DataFrame({'long_coin':coin1_rsi_value,'short_coin':coin2_rsi_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                    ins_4 = pd.DataFrame({'cci_long_coin':coin1_cci_value,'cci_short_coin':coin2_cci_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                    
                    w = 1
                elif price < -0.1 :
                    # coin1为空,coin2为多
                    #ins_1 = pd.DataFrame({'long_coin':coin2_atr_value,'short_coin':coin1_atr_value,'coin1_name':coin1_name,'coin2_name':coin2_name},index=[0])
                    ins_2 = pd.DataFrame({'roc_long_coin':coin2_roc_value,'roc_short_coin':coin1_roc_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                    ins_3 = pd.DataFrame({'long_coin':coin2_rsi_value,'short_coin':coin1_rsi_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                    ins_4 = pd.DataFrame({'cci_long_coin':coin2_cci_value,'cci_short_coin':coin1_cci_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                    w = 1
                else:
                    #ins_1 = pd.DataFrame()
                    ins_2 = pd.DataFrame()
                    ins_3 = pd.DataFrame()
                    ins_4 = pd.DataFrame()
                if w == 1:
                    break
                else:
                    continue
            #art_df = pd.concat([art_df,ins_1])
            roc_df = pd.concat([roc_df,ins_2])
            rsi_df = pd.concat([rsi_df,ins_3])
            cci_df = pd.concat([cci_df,ins_4])

        else:
            # 按照最后一个算，涨多的 多，涨少的为 空

            if last_coin1_price >= last_coin2_price:
                # coin1为多,coin2为空
                #ins_1 = pd.DataFrame({'long_coin':coin1_atr_value,'short_coin':coin2_atr_value,'coin1_name':coin1_name,'coin2_name':coin2_name},index=[0])
                ins_2 = pd.DataFrame({'roc_long_coin':coin1_roc_value,'roc_short_coin':coin2_roc_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                ins_3 = pd.DataFrame({'long_coin':coin1_rsi_value,'short_coin':coin2_rsi_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                ins_4 = pd.DataFrame({'cci_long_coin':coin1_vol_value,'cci_short_coin':coin2_vol_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
            else:
                # coin1为空，coin2为多
                #ins_1 = pd.DataFrame({'long_coin':coin2_atr_value,'short_coin':coin1_atr_value,'coin1_name':coin1_name,'coin2_name':coin2_name},index=[0])
                ins_2 = pd.DataFrame({'roc_long_coin':coin2_roc_value,'roc_short_coin':coin1_roc_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                ins_3 = pd.DataFrame({'long_coin':coin2_rsi_value,'short_coin':coin1_rsi_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
                ins_4 = pd.DataFrame({'cci_long_coin':coin2_vol_value,'cci_short_coin':coin1_vol_value,'coin1_name':coin1_name,'coin2_name':coin2_name,'date':date_interval[1]},index=[0])
            #art_df = pd.concat([art_df,ins_1])
            roc_df = pd.concat([roc_df,ins_2])
            rsi_df = pd.concat([rsi_df,ins_3])
            cci_df = pd.concat([cci_df,ins_4])

        i += 1

        
        
new_df = cci_df.merge(roc_df,how='inner',on=['date','coin1_name','coin2_name'])
new_df['flag1'] = new_df['cci_long_coin'] - new_df['cci_short_coin']
new_df['flag2'] = new_df['roc_long_coin'] - new_df['roc_short_coin']

new_df_1 = new_df[(new_df.flag1>0) &(new_df.flag2>0)]
new_df_2 = new_df[(new_df.flag1<0) &(new_df.flag2<0)]
new_df_3 = new_df[((new_df.flag1>=0) &(new_df.flag2<=0)) | ((new_df.flag1<=0) &(new_df.flag2>=0))]
print(new_df_1)
print(new_df_2)
print(new_df_3)

# =========== 选择 roc 和 cci 同方向的时候，roc的差异
import pandas as pd
from scipy import stats
group1 = new_df_1['roc_long_coin']
group2 = new_df_1['roc_short_coin']
roc_long_mean_1 = np.mean(new_df_1['roc_long_coin'])
roc_short_mean_1 = np.mean(new_df_1['roc_short_coin'])
roc_long_std_1 = np.std(group1)
roc_short_std_1 = np.std(group2)
roc_long_len_1 = len(group1)
roc_short_len_1 = len(group2)
roc_levene_stat_1, roc_levene_p_1 = stats.levene(group1, group2)

if roc_levene_p_1 > 0.05:
    # 方差相等，使用 Student's t-test
    roc_t_stat_1, roc_p_value_1 = stats.ttest_ind(group1, group2, equal_var=True)
else:
    # 方差不等，使用 Welch's t-test
    roc_t_stat_1, roc_p_value_1 = stats.ttest_ind(group1, group2, equal_var=False)

    
import pandas as pd
from scipy import stats
group1 = new_df_2['roc_long_coin']
group2 = new_df_2['roc_short_coin']
roc_long_mean_2 = np.mean(new_df_2['roc_long_coin'])
roc_short_mean_2 = np.mean(new_df_2['roc_short_coin'])
roc_long_std_2 = np.std(group1)
roc_short_std_2 = np.std(group2)
roc_long_len_2 = len(group1)
roc_short_len_2 = len(group2)
roc_levene_stat_2, roc_levene_p_2 = stats.levene(group1, group2)

if roc_levene_p_2 > 0.05:
    # 方差相等，使用 Student's t-test
    roc_t_stat_2, roc_p_value_2 = stats.ttest_ind(group1, group2, equal_var=True)
else:
    # 方差不等，使用 Welch's t-test
    roc_t_stat_2, roc_p_value_2 = stats.ttest_ind(group1, group2, equal_var=False)

    
group1 = new_df_3['roc_long_coin']
group2 = new_df_3['roc_short_coin']
roc_long_mean_3 = np.mean(new_df_3['roc_long_coin'])
roc_short_mean_3 = np.mean(new_df_3['roc_short_coin'])
roc_long_std_3 = np.std(group1)
roc_short_std_3 = np.std(group2)
roc_long_len_3 = len(group1)
roc_short_len_3 = len(group2)
roc_levene_stat_3, roc_levene_p_3 = stats.levene(group1, group2)

if roc_levene_p_3 > 0.05:
    # 方差相等，使用 Student's t-test
    roc_t_stat_3, roc_p_value_3 = stats.ttest_ind(group1, group2, equal_var=True)
else:
    # 方差不等，使用 Welch's t-test
    roc_t_stat_3, roc_p_value_3 = stats.ttest_ind(group1, group2, equal_var=False)



# 创建一个2×4的空表格，初始值为0
table_1 = pd.DataFrame(np.zeros((6, 3)), dtype=str)

# 填充特定值，例如将第一行第一列设置为10
table_1.iloc[0, 0] = 'CCI&ROC>0'
table_1.iloc[0, 1] = 'long'
table_1.iloc[0, 2] = 'short'

table_1.iloc[1, 0] = 'MEAN'
table_1.iloc[1, 1] = round(roc_long_mean_1,5)
table_1.iloc[1, 2] = round(roc_short_mean_1,5)


table_1.iloc[2, 0] = 'STD'
table_1.iloc[2, 1] = round(roc_long_std_1,5)
table_1.iloc[2, 2] = round(roc_short_std_1,5)

table_1.iloc[3, 0] = 'Levene_stat_p'
table_1.iloc[3, 1] = round(roc_levene_stat_1,3)
table_1.iloc[3, 2] = round(roc_levene_p_1,3)

table_1.iloc[4, 0] = 'T_stat_p'
table_1.iloc[4, 1] = round(roc_t_stat_1,3)
table_1.iloc[4, 2] = round(roc_p_value_1,3)


table_1.iloc[5, 0] = 'Sample_number'
table_1.iloc[5, 1] = len(new_df_1)
table_1.iloc[5, 2] = len(new_df_1)

# 创建一个2×4的空表格，初始值为0
table_2 = pd.DataFrame(np.zeros((6, 3)), dtype=str)

# 填充特定值，例如将第一行第一列设置为10
table_2.iloc[0, 0] = 'CCI&ROC<0'
table_2.iloc[0, 1] = 'long'
table_2.iloc[0, 2] = 'short'

table_2.iloc[1, 0] = 'MEAN'
table_2.iloc[1, 1] = round(roc_long_mean_2,5)
table_2.iloc[1, 2] = round(roc_short_mean_2,5)


table_2.iloc[2, 0] = 'STD'
table_2.iloc[2, 1] = round(roc_long_std_2,5)
table_2.iloc[2, 2] = round(roc_short_std_2,5)

table_2.iloc[3, 0] = 'Levene_stat_p'
table_2.iloc[3, 1] = round(roc_levene_stat_2,3)
table_2.iloc[3, 2] = round(roc_levene_p_2,3)

table_2.iloc[4, 0] = 'T_stat_p'
table_2.iloc[4, 1] = round(roc_t_stat_2,3)
table_2.iloc[4, 2] = round(roc_p_value_2,3)

table_2.iloc[5, 0] = 'Sample_number'
table_2.iloc[5, 1] = len(new_df_2)
table_2.iloc[5, 2] = len(new_df_2)
# 创建一个2×4的空表格，初始值为0
table_3 = pd.DataFrame(np.zeros((6, 3)), dtype=str)

# 填充特定值，例如将第一行第一列设置为10
table_3.iloc[0, 0] = 'CCI&ROC<>0'
table_3.iloc[0, 1] = 'long'
table_3.iloc[0, 2] = 'short'

table_3.iloc[1, 0] = 'MEAN'
table_3.iloc[1, 1] = round(roc_long_mean_3,5)
table_3.iloc[1, 2] = round(roc_short_mean_3,5)


table_3.iloc[2, 0] = 'STD'
table_3.iloc[2, 1] = round(roc_long_std_3,5)
table_3.iloc[2, 2] = round(roc_short_std_3,5)

table_3.iloc[3, 0] = 'Levene_stat_p'
table_3.iloc[3, 1] = round(roc_levene_stat_3,3)
table_3.iloc[3, 2] = round(roc_levene_p_3,3)

table_3.iloc[4, 0] = 'T_stat_p'
table_3.iloc[4, 1] = round(roc_t_stat_3,3)
table_3.iloc[4, 2] = round(roc_p_value_3,3)

table_3.iloc[5, 0] = 'Sample_number'
table_3.iloc[5, 1] = len(new_df_3)
table_3.iloc[5, 2] = len(new_df_3)

# 使用pd.concat()沿列方向连接
new_result = pd.concat([table_1, table_2, table_3], axis=1)

import pandas as pd
from scipy import stats
group1 = roc_df['roc_long_coin']
group2 = roc_df['roc_short_coin']
roc_long_mean = np.mean(roc_df['roc_long_coin'])
roc_short_mean = np.mean(roc_df['roc_short_coin'])
roc_long_std = np.std(group1)
roc_short_std = np.std(group2)
roc_levene_stat, roc_levene_p = stats.levene(group1, group2)

if roc_levene_p > 0.05:
    # 方差相等，使用 Student's t-test
    roc_t_stat, roc_p_value = stats.ttest_ind(group1, group2, equal_var=True)
else:
    # 方差不等，使用 Welch's t-test
    roc_t_stat, roc_p_value = stats.ttest_ind(group1, group2, equal_var=False)


group1 = rsi_df['long_coin']
group2 = rsi_df['short_coin']
rsi_long_mean = np.mean(rsi_df['long_coin'])
rsi_short_mean = np.mean(rsi_df['short_coin'])
rsi_long_std = np.std(group1)
rsi_short_std = np.std(group2)
rsi_levene_stat, rsi_levene_p = stats.levene(group1, group2)

if rsi_levene_p > 0.05:
    # 方差相等，使用 Student's t-test
    rsi_t_stat, rsi_p_value = stats.ttest_ind(group1, group2, equal_var=True)
else:
    # 方差不等，使用 Welch's t-test
    rsi_t_stat, rsi_p_value = stats.ttest_ind(group1, group2, equal_var=False)



group1 = cci_df['cci_long_coin']
group2 = cci_df['cci_short_coin']
cci_long_mean = np.mean(cci_df['cci_long_coin'])
cci_short_mean = np.mean(cci_df['cci_short_coin'])
cci_long_std = np.std(group1)
cci_short_std = np.std(group2)
cci_levene_stat, cci_levene_p = stats.levene(group1, group2)

if cci_levene_p > 0.05:
    # 方差相等，使用 Student's t-test
    cci_t_stat, cci_p_value = stats.ttest_ind(group1, group2, equal_var=True)
else:
    # 方差不等，使用 Welch's t-test
    cci_t_stat, cci_p_value = stats.ttest_ind(group1, group2, equal_var=False)


# 创建一个2×4的空表格，初始值为0
table_roc = pd.DataFrame(np.zeros((6, 3)), dtype=str)

# 填充特定值，例如将第一行第一列设置为10
table_roc.iloc[0, 0] = 'ROC'
table_roc.iloc[0, 1] = 'long'
table_roc.iloc[0, 2] = 'short'

table_roc.iloc[1, 0] = 'MEAN'
table_roc.iloc[1, 1] = round(roc_long_mean,5)
table_roc.iloc[1, 2] = round(roc_short_mean,5)


table_roc.iloc[2, 0] = 'STD'
table_roc.iloc[2, 1] = round(roc_long_std,5)
table_roc.iloc[2, 2] = round(roc_short_std,5)

table_roc.iloc[3, 0] = 'Levene_stat_p'
table_roc.iloc[3, 1] = round(roc_levene_stat,3)
table_roc.iloc[3, 2] = round(roc_levene_p,3)

table_roc.iloc[4, 0] = 'T_stat_p'
table_roc.iloc[4, 1] = round(roc_t_stat,3)
table_roc.iloc[4, 2] = round(roc_p_value,3)


table_roc.iloc[5, 0] = 'Sample_number'
table_roc.iloc[5, 1] = len(roc_df)
table_roc.iloc[5, 2] = len(roc_df)

# 创建一个2×4的空表格，初始值为0
table_rsi = pd.DataFrame(np.zeros((6, 3)), dtype=str)

# 填充特定值，例如将第一行第一列设置为10
table_rsi.iloc[0, 0] = 'RSI'
table_rsi.iloc[0, 1] = 'long'
table_rsi.iloc[0, 2] = 'short'

table_rsi.iloc[1, 0] = 'MEAN'
table_rsi.iloc[1, 1] = round(rsi_long_mean,5)
table_rsi.iloc[1, 2] = round(rsi_short_mean,5)


table_rsi.iloc[2, 0] = 'STD'
table_rsi.iloc[2, 1] = round(rsi_long_std,5)
table_rsi.iloc[2, 2] = round(rsi_short_std,5)

table_rsi.iloc[3, 0] = 'Levene_stat_p'
table_rsi.iloc[3, 1] = round(rsi_levene_stat,3)
table_rsi.iloc[3, 2] = round(rsi_levene_p,3)

table_rsi.iloc[4, 0] = 'T_stat_p'
table_rsi.iloc[4, 1] = round(rsi_t_stat,3)
table_rsi.iloc[4, 2] = round(rsi_p_value,3)

table_rsi.iloc[5, 0] = 'Sample_number'
table_rsi.iloc[5, 1] = len(rsi_df)
table_rsi.iloc[5, 2] = len(rsi_df)
# 创建一个2×4的空表格，初始值为0
table_cci = pd.DataFrame(np.zeros((6, 3)), dtype=str)

# 填充特定值，例如将第一行第一列设置为10
table_cci.iloc[0, 0] = 'CCI'
table_cci.iloc[0, 1] = 'long'
table_cci.iloc[0, 2] = 'short'

table_cci.iloc[1, 0] = 'MEAN'
table_cci.iloc[1, 1] = round(cci_long_mean,5)
table_cci.iloc[1, 2] = round(cci_short_mean,5)


table_cci.iloc[2, 0] = 'STD'
table_cci.iloc[2, 1] = round(cci_long_std,5)
table_cci.iloc[2, 2] = round(cci_short_std,5)

table_cci.iloc[3, 0] = 'Levene_stat_p'
table_cci.iloc[3, 1] = round(cci_levene_stat,3)
table_cci.iloc[3, 2] = round(cci_levene_p,3)

table_cci.iloc[4, 0] = 'T_stat_p'
table_cci.iloc[4, 1] = round(cci_t_stat,3)
table_cci.iloc[4, 2] = round(cci_p_value,3)

table_cci.iloc[5, 0] = 'Sample_number'
table_cci.iloc[5, 1] = len(cci_df)
table_cci.iloc[5, 2] = len(cci_df)

# 使用pd.concat()沿列方向连接
test_result = pd.concat([table_roc, table_rsi, table_cci], axis=1)

#print('=============== test_result test_result ===============')
#print(test_result)



# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df
# 读取15分钟数据
btc_data_15m = pd.read_csv('btc_15m_data.csv')
sol_data_15m = pd.read_csv('sol_15m_data.csv')
eth_data_15m = pd.read_csv('eth_15m_data.csv')
xrp_data_15m = pd.read_csv('xrp_15m_data.csv')
doge_data_15m = pd.read_csv('doge_15m_data.csv')

# 读取1分钟数据
btc_data_1m = pd.read_csv('btc_1m_data.csv')
sol_data_1m = pd.read_csv('sol_1m_data.csv')
eth_data_1m = pd.read_csv('eth_1m_data.csv')
xrp_data_1m = pd.read_csv('xrp_1m_data.csv')
doge_data_1m = pd.read_csv('doge_1m_data.csv')

# 把uct8的数据变为uct0的数据
btc_data_15m['date_time'] = btc_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
sol_data_15m['date_time'] = sol_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
eth_data_15m['date_time'] = eth_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
xrp_data_15m['date_time'] = xrp_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
doge_data_15m['date_time'] = doge_data_15m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
btc_data_15m['date'] = btc_data_15m['date_time'].apply(lambda x:x.date())
sol_data_15m['date'] = sol_data_15m['date_time'].apply(lambda x:x.date())
eth_data_15m['date'] = eth_data_15m['date_time'].apply(lambda x:x.date())
xrp_data_15m['date'] = xrp_data_15m['date_time'].apply(lambda x:x.date())
doge_data_15m['date'] = doge_data_15m['date_time'].apply(lambda x:x.date())

btc_data_1m['date_time'] = btc_data_1m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
sol_data_1m['date_time'] = sol_data_1m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
eth_data_1m['date_time'] = eth_data_1m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
xrp_data_1m['date_time'] = xrp_data_1m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
doge_data_1m['date_time'] = doge_data_1m['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
btc_data_1m['date'] = btc_data_1m['date_time'].apply(lambda x:x.date())
sol_data_1m['date'] = sol_data_1m['date_time'].apply(lambda x:x.date())
eth_data_1m['date'] = eth_data_1m['date_time'].apply(lambda x:x.date())
xrp_data_1m['date'] = xrp_data_1m['date_time'].apply(lambda x:x.date())
doge_data_1m['date'] = doge_data_1m['date_time'].apply(lambda x:x.date())

btc_data_15m = btc_data_15m[['date_time','date','close_price','high_price','low_price']]
sol_data_15m = sol_data_15m[['date_time','date','close_price','high_price','low_price']]
eth_data_15m = eth_data_15m[['date_time','date','close_price','high_price','low_price']]
xrp_data_15m = xrp_data_15m[['date_time','date','close_price','high_price','low_price']]
doge_data_15m = doge_data_15m[['date_time','date','close_price','high_price','low_price']]

btc_data_1m = btc_data_1m[['date_time','date','close_price']]
sol_data_1m = sol_data_1m[['date_time','date','close_price']]
eth_data_1m = eth_data_1m[['date_time','date','close_price']]
xrp_data_1m = xrp_data_1m[['date_time','date','close_price']]
doge_data_1m = doge_data_1m[['date_time','date','close_price']]
btc_data_1m.rename(columns={'close_price':'btc_price'},inplace=True)
eth_data_1m.rename(columns={'close_price':'eth_price'},inplace=True)
sol_data_1m.rename(columns={'close_price':'sol_price'},inplace=True)
xrp_data_1m.rename(columns={'close_price':'xrp_price'},inplace=True)
doge_data_1m.rename(columns={'close_price':'doge_price'},inplace=True)


date_list = list(sorted(set(btc_data_15m['date'])))
date_p = str(date_list[1])
data_l = str(date_list[-2])

print(date_p,data_l)
btc_data_15m = btc_data_15m[(btc_data_15m.date>=pd.to_datetime(date_p))&(btc_data_15m.date<=pd.to_datetime(data_l))]
sol_data_15m = sol_data_15m[(sol_data_15m.date>=pd.to_datetime(date_p))&(sol_data_15m.date<=pd.to_datetime(data_l))]
eth_data_15m = eth_data_15m[(eth_data_15m.date>=pd.to_datetime(date_p))&(eth_data_15m.date<=pd.to_datetime(data_l))]
xrp_data_15m = xrp_data_15m[(xrp_data_15m.date>=pd.to_datetime(date_p))&(xrp_data_15m.date<=pd.to_datetime(data_l))]
doge_data_15m = doge_data_15m[(doge_data_15m.date>=pd.to_datetime(date_p))&(doge_data_15m.date<=pd.to_datetime(data_l))]

btc_data_1m = btc_data_1m[(btc_data_1m.date>=pd.to_datetime(date_p))&(btc_data_1m.date<=pd.to_datetime(data_l))]
sol_data_1m = sol_data_1m[(sol_data_1m.date>=pd.to_datetime(date_p))&(sol_data_1m.date<=pd.to_datetime(data_l))]
eth_data_1m = eth_data_1m[(eth_data_1m.date>=pd.to_datetime(date_p))&(eth_data_1m.date<=pd.to_datetime(data_l))]
xrp_data_1m = xrp_data_1m[(xrp_data_1m.date>=pd.to_datetime(date_p))&(xrp_data_1m.date<=pd.to_datetime(data_l))]
doge_data_1m = doge_data_1m[(doge_data_1m.date>=pd.to_datetime(date_p))&(doge_data_1m.date<=pd.to_datetime(data_l))]

date_period = list(sorted(set(btc_data_15m['date'])))
length = len(date_period)

look_df = pd.DataFrame()
i = 0
while i < length-1:
    date_interval = date_period[i:i+2]
    date_0 = date_interval[0]
    date_1 = date_interval[1]
    #print(date_0,date_1)
    
    sub_btc_data_15m = btc_data_15m[btc_data_15m.date==date_0]
    sub_btc_data_15m = sub_btc_data_15m.sort_values(by='date_time')
    sub_sol_data_15m = sol_data_15m[sol_data_15m.date==date_0]
    sub_sol_data_15m = sub_sol_data_15m.sort_values(by='date_time')
    sub_eth_data_15m = eth_data_15m[eth_data_15m.date==date_0]
    sub_eth_data_15m = sub_eth_data_15m.sort_values(by='date_time')
    sub_xrp_data_15m = xrp_data_15m[xrp_data_15m.date==date_0]
    sub_xrp_data_15m = sub_xrp_data_15m.sort_values(by='date_time')
    sub_doge_data_15m = doge_data_15m[doge_data_15m.date==date_0]
    sub_doge_data_15m = sub_doge_data_15m.sort_values(by='date_time')
    
    btc_rsi = calculate_rsi(sub_btc_data_15m)
    btc_rsi = btc_rsi.dropna()
    btc_rsi_value = np.mean(btc_rsi['RSI'])
    
    sol_rsi = calculate_rsi(sub_sol_data_15m)
    sol_rsi = sol_rsi.dropna()
    sol_rsi_value = np.mean(sol_rsi['RSI'])
    
    eth_rsi = calculate_rsi(sub_eth_data_15m)
    eth_rsi = eth_rsi.dropna()
    eth_rsi_value = np.mean(eth_rsi['RSI'])
    
    xrp_rsi = calculate_rsi(sub_xrp_data_15m)
    xrp_rsi = xrp_rsi.dropna()
    xrp_rsi_value = np.mean(xrp_rsi['RSI'])
    
    doge_rsi = calculate_rsi(sub_doge_data_15m)
    doge_rsi = doge_rsi.dropna()
    doge_rsi_value = np.mean(doge_rsi['RSI'])
    
    #print('rsi——value')
    #print(btc_rsi_value,sol_rsi_value,eth_rsi_value,xrp_rsi_value,doge_rsi_value)
    
    btc_roc = calculate_roc(sub_btc_data_15m)
    btc_roc = btc_roc.dropna()
    btc_roc_value = np.mean(btc_roc['ROC'])
    
    sol_roc = calculate_roc(sub_sol_data_15m)
    sol_roc = sol_roc.dropna()
    sol_roc_value = np.mean(sol_roc['ROC'])
    
    eth_roc = calculate_roc(sub_eth_data_15m)
    eth_roc = eth_roc.dropna()
    eth_roc_value = np.mean(eth_roc['ROC'])
    
    xrp_roc = calculate_roc(sub_xrp_data_15m)
    xrp_roc = xrp_roc.dropna()
    xrp_roc_value = np.mean(xrp_roc['ROC'])
    
    doge_roc = calculate_roc(sub_doge_data_15m)
    doge_roc = doge_roc.dropna()
    doge_roc_value = np.mean(doge_roc['ROC'])
    
    
    btc_cci = calculate_cci(sub_btc_data_15m, n=20)
    btc_cci = btc_cci.dropna()
    btc_cci_value = np.mean(btc_cci)
    
    sol_cci = calculate_cci(sub_sol_data_15m, n=20)
    sol_cci = sol_cci.dropna()
    sol_cci_value = np.mean(sol_cci)
    
    eth_cci = calculate_cci(sub_eth_data_15m, n=20)
    eth_cci = eth_cci.dropna()
    eth_cci_value = np.mean(eth_cci)
    
    xrp_cci = calculate_cci(sub_xrp_data_15m, n=20)
    xrp_cci = xrp_cci.dropna()
    xrp_cci_value = np.mean(xrp_cci)
    
    doge_cci = calculate_cci(sub_doge_data_15m, n=20)
    doge_cci = doge_cci.dropna()
    doge_cci_value = np.mean(doge_cci)
    
    #print('roc——value')
    #print(btc_roc_value,sol_roc_value,eth_roc_value,xrp_roc_value,doge_roc_value)
    symbol_list = ['btc','eth','xrp','doge','sol']
    last_df = pd.DataFrame()

    for pair in itertools.combinations(symbol_list, 2):
        coin_1 = pair[0]
        coin_2 = pair[1]
        if coin_1 == 'btc':
            coin1_rsi_value = btc_rsi_value
            coin1_roc_value = btc_roc_value
            coin1_cci_value = btc_cci_value
        elif coin_1 == 'eth':
            coin1_rsi_value = eth_rsi_value
            coin1_roc_value = eth_roc_value
            coin1_cci_value = eth_cci_value
        elif coin_1 == 'xrp':
            coin1_rsi_value = xrp_rsi_value
            coin1_roc_value = xrp_roc_value
            coin1_cci_value = xrp_cci_value
        elif coin_1 == 'sol':
            coin1_rsi_value = sol_rsi_value
            coin1_roc_value = sol_roc_value
            coin1_cci_value = sol_cci_value
        elif coin_1 == 'doge':
            coin1_rsi_value = doge_rsi_value
            coin1_roc_value = doge_roc_value
            coin1_cci_value = doge_cci_value
        else:
            coin1_rsi_value = 0
            coin1_roc_value = 0
        if coin_2 == 'btc':
            coin2_rsi_value = btc_rsi_value
            coin2_roc_value = btc_roc_value
            coin2_cci_value = btc_cci_value
        elif coin_2 == 'eth':
            coin2_rsi_value = eth_rsi_value
            coin2_roc_value = eth_roc_value
            coin2_cci_value = eth_cci_value
        elif coin_2 == 'xrp':
            coin2_rsi_value = xrp_rsi_value
            coin2_roc_value = xrp_roc_value
            coin2_cci_value = xrp_cci_value
        elif coin_2 == 'sol':
            coin2_rsi_value = sol_rsi_value
            coin2_roc_value = sol_roc_value
            coin2_cci_value = sol_cci_value
        elif coin_2 == 'doge':
            coin2_rsi_value = doge_rsi_value
            coin2_roc_value = doge_roc_value
            coin2_cci_value = doge_cci_value
        else:
            coin2_rsi_value = 0
            coin2_roc_value = 0
            
        ins = pd.DataFrame({'coin_1_name':coin_1,'coin_2_name':coin_2,'coin1_rsi_value':coin1_rsi_value,'coin2_rsi_value':coin2_rsi_value,'rsi_d_abs':(coin1_rsi_value-coin2_rsi_value)/np.abs(coin1_rsi_value-coin2_rsi_value),'coin1_roc_value':coin1_roc_value,'coin2_roc_value':coin2_roc_value,'roc_d':coin1_roc_value-coin2_roc_value,'roc_d_abs':(coin1_roc_value-coin2_roc_value)/np.abs(coin1_roc_value-coin2_roc_value),'cci_d':coin1_cci_value-coin2_cci_value,'cci_d_abs':(coin1_cci_value-coin2_cci_value)/np.abs(coin1_cci_value-coin2_cci_value)},index=[0])
        last_df = pd.concat([last_df,ins])
    #print('last_df')
    #print(last_df)
    
    last_df_1 = last_df[(last_df.cci_d_abs==1) &(last_df.roc_d_abs==1)]
    last_df_2 = last_df[(last_df.cci_d_abs==-1)&(last_df.roc_d_abs==-1)]
    #last_df_1 = last_df[(last_df.roc_d_abs==1)]
    #last_df_2 = last_df[(last_df.roc_d_abs==-1)]
    if len(last_df_1) > 0 and len(last_df_2)>0:
        max_abs_value_1 = last_df_1['roc_d'].abs().max()
        max_abs_value_2 = last_df_2['roc_d'].abs().max()
        if max_abs_value_1 >= max_abs_value_2:
            # 选1 最大
            last_df_1['flag'] = last_df_1['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_1 else 0) 
            sub_last_df_1 = last_df_1[last_df_1.flag==1]
            sub_last_df_1 = sub_last_df_1.reset_index(drop=True)
            
            # 做多coin1，做空coin2
            coin_long = sub_last_df_1['coin_1_name'][0] 
            coin_short = sub_last_df_1['coin_2_name'][0] 
        else:
            # 选2 最小
            last_df_2['flag'] = last_df_2['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_2 else 0)
            sub_last_df_2 = last_df_2[last_df_2.flag==1]
            sub_last_df_2 = sub_last_df_2.reset_index(drop=True)
            
            # 做多coin2，做空coin1
            coin_long = sub_last_df_2['coin_2_name'][0] 
            coin_short = sub_last_df_2['coin_1_name'][0]    
            
    elif len(last_df_1) > 0 and len(last_df_2)==0:
        max_abs_value_1 = last_df_1['roc_d'].abs().max()

        last_df_1['flag'] = last_df_1['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_1 else 0) 
        sub_last_df_1 = last_df_1[last_df_1.flag==1]
        sub_last_df_1 = sub_last_df_1.reset_index(drop=True)

        # 做多coin1，做空coin2
        coin_long = sub_last_df_1['coin_1_name'][0] 
        coin_short = sub_last_df_1['coin_2_name'][0] 

    elif len(last_df_1) == 0 and len(last_df_2)>0:
        max_abs_value_2 = last_df_2['roc_d'].abs().max()
 
        last_df_2['flag'] = last_df_2['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_2 else 0)
        sub_last_df_2 = last_df_2[last_df_2.flag==1]
        sub_last_df_2 = sub_last_df_2.reset_index(drop=True)

        # 做多coin2，做空coin1
        coin_long = sub_last_df_2['coin_2_name'][0] 
        coin_short = sub_last_df_2['coin_1_name'][0] 
    else:
        coin_long = None
        coin_short = None

    # =========================== 进行正式的跑数据 ============================
    sub_btc_data_1m = btc_data_1m[btc_data_1m.date==date_1]
    sub_btc_data_1m = sub_btc_data_1m.sort_values(by='date_time')
    sub_btc_data_1m = sub_btc_data_1m.reset_index(drop=True)
    sub_eth_data_1m = eth_data_1m[eth_data_1m.date==date_1]
    sub_eth_data_1m = sub_eth_data_1m.sort_values(by='date_time')
    sub_eth_data_1m = sub_eth_data_1m.reset_index(drop=True)
    sub_sol_data_1m = sol_data_1m[sol_data_1m.date==date_1]
    sub_sol_data_1m = sub_sol_data_1m.sort_values(by='date_time')
    sub_sol_data_1m = sub_sol_data_1m.reset_index(drop=True)
    sub_xrp_data_1m = xrp_data_1m[xrp_data_1m.date==date_1]
    sub_xrp_data_1m = sub_xrp_data_1m.sort_values(by='date_time')
    sub_xrp_data_1m = sub_xrp_data_1m.reset_index(drop=True)
    sub_doge_data_1m = doge_data_1m[doge_data_1m.date==date_1]
    sub_doge_data_1m = sub_doge_data_1m.sort_values(by='date_time')
    sub_doge_data_1m = sub_doge_data_1m.reset_index(drop=True) 
    #print('sub_btc_data_1m')
    #print(sub_btc_data_1m)
    print(coin_long,coin_short)
    if coin_long == 'btc':
        coin1_data = sub_btc_data_1m
    elif coin_long == 'eth':
        coin1_data = sub_eth_data_1m
    elif coin_long == 'xrp':
        coin1_data = sub_xrp_data_1m
    elif coin_long == 'sol':
        coin1_data = sub_sol_data_1m
    elif coin_long == 'doge':
        coin1_data = sub_doge_data_1m
    else:
        coin1_data = None
    if coin_short == 'btc':
        coin2_data = sub_btc_data_1m
    elif coin_short == 'eth':
        coin2_data = sub_eth_data_1m
    elif coin_short == 'xrp':
        coin2_data = sub_xrp_data_1m
    elif coin_short == 'sol':
        coin2_data = sub_sol_data_1m
    elif coin_short == 'doge':
        coin2_data = sub_doge_data_1m
    else:
        coin2_data = None
        
    combine_data =  coin1_data.merge(coin2_data,how='left',on=['date_time','date'])
    coin1_price = coin_long + '_price'
    coin2_price = coin_short + '_price'
    # 获取第一行的值
    coin1_first_value = combine_data[coin1_price].iloc[0]
    # 计算每行相对于第一行的变化率
    combine_data['coin1_change_rate'] = (combine_data[coin1_price] - coin1_first_value) / coin1_first_value
    coin2_first_value = combine_data[coin2_price].iloc[0]
    # 计算每行相对于第一行的变化率
    combine_data['coin2_change_rate'] = (combine_data[coin2_price] - coin2_first_value) / coin2_first_value

    combine_data['change'] = combine_data['coin1_change_rate'] - combine_data['coin2_change_rate']
    combine_data = combine_data.dropna()
    combine_data = combine_data.reset_index(drop=True)
    #print('combine_data')
    #print(combine_data)
    p = 0
    q = 0
    for w in range(len(combine_data)):
        #print(w,coin_long,coin_short,value)
        value = combine_data['change'][w]
        if value > 0.15:
            # 成功
            open_df = pd.DataFrame({'coin_long':coin_long,'coin_short':coin_short,'value':value,'date':date_1},index=[0])
            #print(open_df)
            p = 1 
        elif value < -0.15:
            # 失败
            open_df = pd.DataFrame({'coin_long':coin_long,'coin_short':coin_short,'value':value,'date':date_1},index=[0])
            #print(open_df)
            p = 1
            
        if p == 0 and q < len(combine_data) - 1:
            q += 1
            continue
        elif p == 0 and q == len(combine_data) - 1:
            open_df = pd.DataFrame({'coin_long':coin_long,'coin_short':coin_short,'value':value,'date':date_1},index=[0])
            #print(open_df)
        else:
            break
        
    look_df = pd.concat([look_df,open_df])
    #print('look_df')
    #print(look_df)
    i += 1


# 计算数值列的平均值和中位数
sum_values = look_df.sum(numeric_only=True)
mean_values = look_df.mean(numeric_only=True)
median_values = look_df.median(numeric_only=True)
max_values = look_df.max(numeric_only=True)
min_values = look_df.min(numeric_only=True)
success_values = round(len(look_df[look_df.value>0])/len(look_df),4)
action_values = len(look_df)
# 为非数值列填充占位符
for col in look_df.columns:
    if col not in mean_values.index:
        sum_values[col] = '—'
        mean_values[col] = '—'
        median_values[col] = '—'
        max_values[col] = '—'
        min_values[col] = '—'

# 添加标识列
sum_values['Name'] = 'Sum'
mean_values['Name'] = 'Mean'
median_values['Name'] = 'Median'
max_values['Name'] = 'Max'
min_values['Name'] = 'Min'

# 将新行添加到原DataFrame
look_df = pd.concat([look_df,sum_values.to_frame().T, mean_values.to_frame().T, median_values.to_frame().T, max_values.to_frame().T, min_values.to_frame().T], ignore_index=True)

table_model_1 = look_df
test_result.columns = test_result.iloc[0]

# 删除第一行
test_result = test_result[1:].reset_index(drop=True)

# ================================================================================== 第二个模型 =========================================================
import importlib
import sys
import os
import urllib
import requests
import base64
import json

import time
import pandas as pd
import numpy as np
import random
import hmac
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')

btc_data = pd.read_csv('btc_15m_data.csv')
sol_data = pd.read_csv('sol_15m_data.csv')
eth_data = pd.read_csv('eth_15m_data.csv')
xrp_data = pd.read_csv('xrp_15m_data.csv')
doge_data = pd.read_csv('doge_15m_data.csv')

btc_data['date_time'] = btc_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
eth_data['date_time'] = eth_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
sol_data['date_time'] = sol_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
xrp_data['date_time'] = xrp_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
doge_data['date_time'] = doge_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))

btc_data = btc_data.drop('formatted_time', axis=1)
eth_data = eth_data.drop('formatted_time', axis=1)
sol_data = sol_data.drop('formatted_time', axis=1)
xrp_data = xrp_data.drop('formatted_time', axis=1)
doge_data = doge_data.drop('formatted_time', axis=1)

btc_data['date'] = btc_data['date_time'].apply(lambda x:x.date())
btc_data['hour'] = btc_data['date_time'].apply(lambda x:x.hour)
btc_data['minutes'] = btc_data['date_time'].apply(lambda x:x.minute)

eth_data['date'] = eth_data['date_time'].apply(lambda x:x.date())
eth_data['hour'] = eth_data['date_time'].apply(lambda x:x.hour)
eth_data['minutes'] = eth_data['date_time'].apply(lambda x:x.minute)

sol_data['date'] = sol_data['date_time'].apply(lambda x:x.date())
sol_data['hour'] = sol_data['date_time'].apply(lambda x:x.hour)
sol_data['minutes'] = sol_data['date_time'].apply(lambda x:x.minute)

xrp_data['date'] = xrp_data['date_time'].apply(lambda x:x.date())
xrp_data['hour'] = xrp_data['date_time'].apply(lambda x:x.hour)
xrp_data['minutes'] = xrp_data['date_time'].apply(lambda x:x.minute)

doge_data['date'] = doge_data['date_time'].apply(lambda x:x.date())
doge_data['hour'] = doge_data['date_time'].apply(lambda x:x.hour)
doge_data['minutes'] = doge_data['date_time'].apply(lambda x:x.minute)

date_list = list(sorted(set(btc_data['date'])))
date_p = str(date_list[1])
data_l = str(date_list[-2])

print(date_p,data_l)
btc_data_15m = btc_data[(btc_data.date>=pd.to_datetime(date_p))&(btc_data.date<=pd.to_datetime(data_l))]
sol_data_15m = sol_data[(sol_data.date>=pd.to_datetime(date_p))&(sol_data.date<=pd.to_datetime(data_l))]
eth_data_15m = eth_data[(eth_data.date>=pd.to_datetime(date_p))&(eth_data.date<=pd.to_datetime(data_l))]
xrp_data_15m = xrp_data[(xrp_data.date>=pd.to_datetime(date_p))&(xrp_data.date<=pd.to_datetime(data_l))]
doge_data_15m = doge_data[(doge_data.date>=pd.to_datetime(date_p))&(doge_data.date<=pd.to_datetime(data_l))]
btc_data_15m = btc_data_15m.sort_values(by='date_time')
sol_data_15m = sol_data_15m.sort_values(by='date_time')
eth_data_15m = eth_data_15m.sort_values(by='date_time')
xrp_data_15m = xrp_data_15m.sort_values(by='date_time')
doge_data_15m = doge_data_15m.sort_values(by='date_time')


# =================================== 计算cci，roc，rsi的函数定义 =====================
# 计算平均真实波动性
import pandas_ta as ta
def calculate_cci(data, n=20):
    """
    计算商品通道指数（CCI）。

    参数：
    data : pd.DataFrame
        包含 'high'、'low' 和 'close' 列的 DataFrame。
    n : int
        计算 CCI 的周期，默认值为 20。

    返回：
    pd.Series
        计算得到的 CCI 值。
    """
    # 计算典型价格（TP）
    tp = (data['high_price'] + data['low_price'] + data['close_price']) / 3

    # 计算 TP 的 N 日简单移动平均（SMA）
    sma_tp = tp.rolling(window=n, min_periods=1).mean()

    # 计算平均绝对偏差（Mean Deviation，MD）
    md = tp.rolling(window=n, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)

    # 计算 CCI
    cci = (tp - sma_tp) / (0.015 * md)

    return cci

# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

# 计算价格变化率（Rate of Change）
def calculate_vol(df, column='usdt_volumn', window=14):
    df['VOL'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df

def calculate_price_change(df):
    df = df.sort_values(by='date_time')
    df = df.reset_index(drop=True)
    first_value = df['open_price'][0]
    last_value = df['close_price'][len(df)-1]
    price_change = (last_value-first_value)/first_value
    return price_change


# ==================================== 形成训练集 ====================================

date_period = list(sorted(set(btc_data_15m['date'])))
length = len(date_period)
last_df = pd.DataFrame()
i = 0
while i < length-1:
    date_interval = date_period[i:i+2]
    date_0 = date_interval[0]
    date_1 = date_interval[1]
    print(date_0,date_1)
    # =======================。前一天的参数
    btc_data_15m_sample = btc_data_15m[btc_data_15m.date==date_0]
    btc_data_15m_sample = btc_data_15m_sample.sort_values(by='date_time')
    sol_data_15m_sample = sol_data_15m[sol_data_15m.date==date_0]
    sol_data_15m_sample = sol_data_15m_sample.sort_values(by='date_time')
    eth_data_15m_sample = eth_data_15m[eth_data_15m.date==date_0]
    eth_data_15m_sample = eth_data_15m_sample.sort_values(by='date_time')
    xrp_data_15m_sample = xrp_data_15m[xrp_data_15m.date==date_0]
    xrp_data_15m_sample = xrp_data_15m_sample.sort_values(by='date_time')
    doge_data_15m_sample = doge_data_15m[doge_data_15m.date==date_0]
    doge_data_15m_sample = doge_data_15m_sample.sort_values(by='date_time')
    
    # ======================== 后一天的结果
    btc_data_15m_result = btc_data_15m[btc_data_15m.date==date_1]
    btc_data_15m_result = btc_data_15m_result.sort_values(by='date_time')
    btc_data_15m_result = btc_data_15m_result.reset_index(drop=True)
    sol_data_15m_result = sol_data_15m[sol_data_15m.date==date_1]
    sol_data_15m_result = sol_data_15m_result.sort_values(by='date_time')
    sol_data_15m_result = sol_data_15m_result.reset_index(drop=True)
    eth_data_15m_result = eth_data_15m[eth_data_15m.date==date_1]
    eth_data_15m_result = eth_data_15m_result.sort_values(by='date_time')
    eth_data_15m_result = eth_data_15m_result.reset_index(drop=True)
    xrp_data_15m_result = xrp_data_15m[xrp_data_15m.date==date_1]
    xrp_data_15m_result = xrp_data_15m_result.sort_values(by='date_time')
    xrp_data_15m_result = xrp_data_15m_result.reset_index(drop=True)
    doge_data_15m_result = doge_data_15m[doge_data_15m.date==date_1]
    doge_data_15m_result = doge_data_15m_result.sort_values(by='date_time')
    doge_data_15m_result = doge_data_15m_result.reset_index(drop=True)
    
    
    #print(btc_data_15m_sample)
    #print(btc_data_15m_result)
    
    #print(eth_data_15m_sample)
    #print(eth_data_15m_result)
    
    btc_rsi = calculate_rsi(btc_data_15m_sample)
    btc_rsi = btc_rsi.dropna()
    btc_rsi_value = np.mean(btc_rsi['RSI'])
    
    sol_rsi = calculate_rsi(sol_data_15m_sample)
    sol_rsi = sol_rsi.dropna()
    sol_rsi_value = np.mean(sol_rsi['RSI'])
    
    eth_rsi = calculate_rsi(eth_data_15m_sample)
    eth_rsi = eth_rsi.dropna()
    eth_rsi_value = np.mean(eth_rsi['RSI'])
    
    xrp_rsi = calculate_rsi(xrp_data_15m_sample)
    xrp_rsi = xrp_rsi.dropna()
    xrp_rsi_value = np.mean(xrp_rsi['RSI'])
    
    doge_rsi = calculate_rsi(doge_data_15m_sample)
    doge_rsi = doge_rsi.dropna()
    doge_rsi_value = np.mean(doge_rsi['RSI'])
    
    #print('rsi——value')
    #print(btc_rsi_value,sol_rsi_value,eth_rsi_value,xrp_rsi_value,doge_rsi_value)
    
    btc_roc = calculate_roc(btc_data_15m_sample)
    btc_roc = btc_roc.dropna()
    btc_roc_value = np.mean(btc_roc['ROC'])
    
    sol_roc = calculate_roc(sol_data_15m_sample)
    sol_roc = sol_roc.dropna()
    sol_roc_value = np.mean(sol_roc['ROC'])
    
    eth_roc = calculate_roc(eth_data_15m_sample)
    eth_roc = eth_roc.dropna()
    eth_roc_value = np.mean(eth_roc['ROC'])
    
    xrp_roc = calculate_roc(xrp_data_15m_sample)
    xrp_roc = xrp_roc.dropna()
    xrp_roc_value = np.mean(xrp_roc['ROC'])
    
    doge_roc = calculate_roc(doge_data_15m_sample)
    doge_roc = doge_roc.dropna()
    doge_roc_value = np.mean(doge_roc['ROC'])
    
    
    btc_cci = calculate_cci(btc_data_15m_sample, n=20)
    btc_cci = btc_cci.dropna()
    btc_cci_value = np.mean(btc_cci)
    
    sol_cci = calculate_cci(sol_data_15m_sample, n=20)
    sol_cci = sol_cci.dropna()
    sol_cci_value = np.mean(sol_cci)
    
    eth_cci = calculate_cci(eth_data_15m_sample, n=20)
    eth_cci = eth_cci.dropna()
    eth_cci_value = np.mean(eth_cci)
    
    xrp_cci = calculate_cci(xrp_data_15m_sample, n=20)
    xrp_cci = xrp_cci.dropna()
    xrp_cci_value = np.mean(xrp_cci)
    
    doge_cci = calculate_cci(doge_data_15m_sample, n=20)
    doge_cci = doge_cci.dropna()
    doge_cci_value = np.mean(doge_cci)
    
    #print('roc——value')
    #print(btc_roc_value,sol_roc_value,eth_roc_value,xrp_roc_value,doge_roc_value)
    #symbol_list = ['btc','eth','xrp','doge','sol']
    symbol_list = ['btc','eth','xrp','doge','sol']


    for pair in itertools.combinations(symbol_list, 2):
    #for pair in symbol_list:
        coin_1 = pair[0]
        coin_2 =  pair[1]
        if coin_1 == 'btc':
            coin1_rsi_value = btc_rsi_value
            coin1_roc_value = btc_roc_value
            coin1_cci_value = btc_cci_value
            coin1_rs_value = calculate_price_change(btc_data_15m_sample)
            coin1_first_value = btc_data_15m_result['open_price'].iloc[0]
            coin1_price_change = (btc_data_15m_result['close_price'][len(btc_data_15m_result)-1]- coin1_first_value) / coin1_first_value
        elif coin_1 == 'eth':
            coin1_rsi_value = eth_rsi_value
            coin1_roc_value = eth_roc_value
            coin1_cci_value = eth_cci_value
            coin1_rs_value = calculate_price_change(eth_data_15m_sample)
            coin1_first_value = eth_data_15m_result['open_price'].iloc[0]
            coin1_price_change = (eth_data_15m_result['close_price'][len(eth_data_15m_result)-1]- coin1_first_value) / coin1_first_value
        elif coin_1 == 'xrp':
            coin1_rsi_value = xrp_rsi_value
            coin1_roc_value = xrp_roc_value
            coin1_cci_value = xrp_cci_value
            coin1_rs_value = calculate_price_change(xrp_data_15m_sample)
            coin1_first_value = xrp_data_15m_result['open_price'].iloc[0]
            coin1_price_change = (xrp_data_15m_result['close_price'][len(xrp_data_15m_result)-1]- coin1_first_value) / coin1_first_value
        elif coin_1 == 'sol':
            coin1_rsi_value = sol_rsi_value
            coin1_roc_value = sol_roc_value
            coin1_cci_value = sol_cci_value
            coin1_rs_value = calculate_price_change(sol_data_15m_sample)
            coin1_first_value = sol_data_15m_result['open_price'].iloc[0]
            coin1_price_change = (sol_data_15m_result['close_price'][len(sol_data_15m_result)-1]- coin1_first_value) / coin1_first_value
        elif coin_1 == 'doge':
            coin1_rsi_value = doge_rsi_value
            coin1_roc_value = doge_roc_value
            coin1_cci_value = doge_cci_value
            coin1_rs_value = calculate_price_change(doge_data_15m_sample)
            coin1_first_value = doge_data_15m_result['open_price'].iloc[0]
            coin1_price_change = (doge_data_15m_result['close_price'][len(doge_data_15m_result)-1]- coin1_first_value) / coin1_first_value
        else:
            coin1_rsi_value = 0
            coin1_roc_value = 0
        if coin_2 == 'btc':
            coin2_rsi_value = btc_rsi_value
            coin2_roc_value = btc_roc_value
            coin2_cci_value = btc_cci_value
            coin2_rs_value = calculate_price_change(btc_data_15m_sample)
            coin2_first_value = btc_data_15m_result['open_price'].iloc[0]
            coin2_price_change = (btc_data_15m_result['close_price'][len(btc_data_15m_result)-1]- coin2_first_value) / coin2_first_value   
        elif coin_2 == 'eth':
            coin2_rsi_value = eth_rsi_value
            coin2_roc_value = eth_roc_value
            coin2_cci_value = eth_cci_value
            coin2_rs_value = calculate_price_change(eth_data_15m_sample)
            coin2_first_value = eth_data_15m_result['open_price'].iloc[0]
            coin2_price_change = (eth_data_15m_result['close_price'][len(eth_data_15m_result)-1]- coin2_first_value) / coin2_first_value
        elif coin_2 == 'xrp':
            coin2_rsi_value = xrp_rsi_value
            coin2_roc_value = xrp_roc_value
            coin2_cci_value = xrp_cci_value
            coin2_rs_value = calculate_price_change(xrp_data_15m_sample)
            coin2_first_value = xrp_data_15m_result['open_price'].iloc[0]
            coin2_price_change = (xrp_data_15m_result['close_price'][len(xrp_data_15m_result)-1]- coin2_first_value) / coin2_first_value
        elif coin_2 == 'sol':
            coin2_rsi_value = sol_rsi_value
            coin2_roc_value = sol_roc_value
            coin2_cci_value = sol_cci_value
            coin2_rs_value = calculate_price_change(sol_data_15m_sample)
            coin2_first_value = sol_data_15m_result['open_price'].iloc[0]
            coin2_price_change = (sol_data_15m_result['close_price'][len(sol_data_15m_result)-1]- coin2_first_value) / coin2_first_value
        elif coin_2 == 'doge':
            coin2_rsi_value = doge_rsi_value
            coin2_roc_value = doge_roc_value
            coin2_cci_value = doge_cci_value
            coin2_rs_value = calculate_price_change(doge_data_15m_sample)
            coin2_first_value = doge_data_15m_result['open_price'].iloc[0]
            coin2_price_change = (doge_data_15m_result['close_price'][len(doge_data_15m_result)-1]- coin2_first_value) / coin2_first_value
        else:
            coin2_rsi_value = 0
            coin2_roc_value = 0
            
        ins = pd.DataFrame({'date':date_1,'coin_1_name':coin_1,'coin_2_name':coin_2,'coin_1_rs':coin1_rs_value,'coin_2_rs':coin2_rs_value,'rs_d':coin1_rs_value-coin2_rs_value,'coin1_rsi_value':coin1_rsi_value,'coin2_rsi_value':coin2_rsi_value,'coin1_roc_value':coin1_roc_value,'coin2_roc_value':coin2_roc_value,'coin1_cci_value':coin1_cci_value,'coin2_cci_value':coin2_cci_value,'price_change':coin1_price_change-coin2_price_change,'rsi_d':coin1_rsi_value-coin2_rsi_value,'roc_d':coin1_roc_value-coin2_roc_value,'cci_d':coin1_cci_value-coin2_cci_value},index=[0])
        #print(ins)
        last_df = pd.concat([last_df,ins])
    i += 1

import pickle
features = ['rsi_d', 'roc_d', 'cci_d', 'rs_d']
X_test = last_df[features]
print(X_test)


with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
# 预测均值
y_pred = loaded_model.predict(X_test)


last_df['y_pred'] = y_pred
date_period = list(sorted(set(last_df['date'])))
table_model_2 = pd.DataFrame()
for date in date_period:#[pd.to_datetime('2025-01-20'),pd.to_datetime('2025-01-22')]:
    ins = last_df[last_df.date==date]
    sub_ins = ins[ins.y_pred==np.max(ins['y_pred'])]
    sub_ins = sub_ins.reset_index(drop=True)
    if sub_ins['y_pred'][0]>0:
        coin_long = sub_ins['coin_2_name'][0]
        coin_short = sub_ins['coin_1_name'][0]
        value = -sub_ins['price_change'][0]
        date = sub_ins['date'][0]
        df = pd.DataFrame({'date':date,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])
    else:
        coin_long = sub_ins['coin_1_name'][0]
        coin_short = sub_ins['coin_2_name'][0]
        value = sub_ins['price_change'][0]
        date = sub_ins['date'][0]
        df = pd.DataFrame({'date':date,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])        
    table_model_2 = pd.concat([table_model_2,df])

action_values_model2 = len(table_model_2)
success_values_model2 = len(table_model_2[table_model_2.value>0])/len(table_model_2)
return_values_model2 = np.sum(table_model_2['value'])


# ============================================================ 模型3 ==========================================================
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
# 转换时间戳（秒转换为毫秒）
def to_milliseconds(timestamp):
    return int(timestamp * 1000)

# 获取当前时间的 Unix 时间戳（毫秒）
def current_timestamp():
    return int(time.time() * 1000)

# 获取 3 年前的时间戳（毫秒）
def get_three_years_ago_timestamp():
    three_years_in_seconds = (2 * 365 +60) * 24 * 60 * 60  # 3年 = 3 * 365 * 24 * 60 * 60 秒
    return to_milliseconds(time.time() - three_years_in_seconds)

# 获取资金费率
def get_funding_rates(symbol, start_time, end_time, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        'symbol': symbol,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return []

# 获取近3年内的资金费率数据
def get_funding_rates_for_three_years(symbol):
    current_time = current_timestamp()  # 当前时间
    three_years_ago = get_three_years_ago_timestamp()  # 3年前的时间
    
    all_funding_rates = []  # 用于存储所有资金费率数据
    start_time = three_years_ago
    
    # 逐步请求每1000条资金费率数据，直到获取到当前时间为止
    while start_time < current_time:
        end_time = start_time + 100 * 86400000  # 每次请求1天的数据（86400000 毫秒 = 1天）
        
        # 请求资金费率数据
        funding_rates = get_funding_rates(symbol, start_time, end_time)
        
        if funding_rates:
            all_funding_rates.extend(funding_rates)
        
        start_time = end_time  # 更新下一批的起始时间
        time.sleep(1)
    
    return all_funding_rates

# 示例：获取近3年的BTCUSDT资金费率数据
symbol_list = ['BTCUSDT','ETHUSDT','DOGEUSDT','SOLUSDT','XRPUSDT','LTCUSDT','ADAUSDT']
last_df = pd.DataFrame()
for symbol in symbol_list:
    print(symbol)
    funding_rates = get_funding_rates_for_three_years(symbol)

    # 打印部分结果
    for entry in funding_rates:  # 打印前10条数据
        ins = pd.DataFrame({'symbol':entry['symbol'], 'rate':float(entry['fundingRate']), 'time': entry['fundingTime']},index=[0])
        last_df = pd.concat([last_df,ins])
from datetime import datetime

last_df['date_time'] = last_df['time'].apply(lambda x: datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S') )
last_df.to_csv('fund_rate.csv')
import ccxt
import pandas as pd
import time

# 连接 Binance
exchange = ccxt.binance()

symbol_list = ['BTC/USDT','ETH/USDT','SOL/USDT','XRP/USDT','DOGE/USDT','LTC/USDT','ADA/USDT']
# 设置参数

for symbol in symbol_list:
    #symbol = 'BTC/USDT'  # 交易对
    timeframe = '1d'  # 时间周期
    since = exchange.parse8601('2022-11-01T00:00:00Z')  # 从 2 年前开始
    limit = 500  # 每次最多获取 500 条数据

    # 获取 K 线数据
    all_ohlcv = []
    while since < exchange.milliseconds():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

        if not ohlcv:
            break  # 如果没有更多数据，停止
        all_ohlcv.extend(ohlcv)

        since = ohlcv[-1][0] + 1  # 继续获取下一批数据
        time.sleep(1)  # 避免 API 速率限制

    # 转换为 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 按时间排序
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    name = symbol.split('/')[0]+'1d_data.csv'
    df.to_csv(name)
from datetime import datetime,timedelta
import numpy as np
btc_data = pd.read_csv('BTC1d_data.csv')
sol_data = pd.read_csv('SOL1d_data.csv')
eth_data = pd.read_csv('ETH1d_data.csv')
xrp_data = pd.read_csv('XRP1d_data.csv')
doge_data = pd.read_csv('DOGE1d_data.csv')
ltc_data = pd.read_csv('LTC1d_data.csv')
ada_data = pd.read_csv('ADA1d_data.csv')

btc_data['date_time'] = btc_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
eth_data['date_time'] = eth_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
sol_data['date_time'] = sol_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
xrp_data['date_time'] = xrp_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
doge_data['date_time'] = doge_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
ltc_data['date_time'] = ltc_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
ada_data['date_time'] = ada_data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))


btc_data = btc_data.sort_values(by='date_time')
eth_data = eth_data.sort_values(by='date_time')
sol_data = sol_data.sort_values(by='date_time')
xrp_data = xrp_data.sort_values(by='date_time')
doge_data = doge_data.sort_values(by='date_time')
ltc_data = ltc_data.sort_values(by='date_time')
ada_data = ada_data.sort_values(by='date_time')

fund_rate = pd.read_csv('fund_rate.csv')
fund_rate['date_time'] = fund_rate['date_time'].apply(lambda x:pd.to_datetime(x[0:11]))

btc_fund = fund_rate[fund_rate.symbol=='BTCUSDT']
eth_fund = fund_rate[fund_rate.symbol=='ETHUSDT']
xrp_fund = fund_rate[fund_rate.symbol=='XRPUSDT']
doge_fund = fund_rate[fund_rate.symbol=='DOGEUSDT']
sol_fund = fund_rate[fund_rate.symbol=='SOLUSDT']
ltc_fund = fund_rate[fund_rate.symbol=='LTCUSDT']
ada_fund = fund_rate[fund_rate.symbol=='ADAUSDT']

def calculate_price_change(df):
    df = df.sort_values(by='date_time')
    df = df.reset_index(drop=True)
    first_value = df['open'][0]
    last_value = df['close'][len(df)-1]
    price_change = (last_value-first_value)/first_value
    return price_change
import itertools
date_period = list(sorted(set(btc_data['date_time'])))[0:-1]
length = len(date_period)

look_df = pd.DataFrame()
i = 0
while i < length-1:
    try:
        date_interval = date_period[i:i+5]
        date_s = date_interval[0]
        date_l = date_interval[3]
        date_t = date_interval[4]
        print(date_s,date_l,date_t)

        sub_btc_fund = btc_fund[(btc_fund.date_time>=date_s)&(btc_fund.date_time<=date_l)]
        sub_eth_fund = eth_fund[(eth_fund.date_time>=date_s)&(eth_fund.date_time<=date_l)]
        sub_xrp_fund = xrp_fund[(xrp_fund.date_time>=date_s)&(xrp_fund.date_time<=date_l)]
        sub_doge_fund = doge_fund[(doge_fund.date_time>=date_s)&(doge_fund.date_time<=date_l)]
        sub_sol_fund = sol_fund[(sol_fund.date_time>=date_s)&(sol_fund.date_time<=date_l)]
        sub_ltc_fund = ltc_fund[(ltc_fund.date_time>=date_s)&(ltc_fund.date_time<=date_l)]
        sub_ada_fund = ada_fund[(ada_fund.date_time>=date_s)&(ada_fund.date_time<=date_l)]

        btc_res = btc_data[btc_data.date_time==date_t]
        sol_res = sol_data[sol_data.date_time==date_t]
        eth_res = eth_data[eth_data.date_time==date_t]
        xrp_res = xrp_data[xrp_data.date_time==date_t]
        doge_res = doge_data[doge_data.date_time==date_t]
        ltc_res = ltc_data[ltc_data.date_time==date_t]
        ada_res = ada_data[ada_data.date_time==date_t]

        #print(sub_btc_data)
        #print(btc_res)

        btc_rate = np.mean(sub_btc_fund['rate'])
        btc_std = np.std(sub_btc_fund['rate'])
        btc_price_change = calculate_price_change(btc_res)

        eth_rate = np.mean(sub_eth_fund['rate'])
        eth_std = np.std(sub_eth_fund['rate'])
        eth_price_change = calculate_price_change(eth_res)

        xrp_rate = np.mean(sub_xrp_fund['rate'])
        xrp_std = np.std(sub_xrp_fund['rate'])
        xrp_price_change = calculate_price_change(xrp_res)

        doge_rate = np.mean(sub_doge_fund['rate'])
        doge_std = np.std(sub_doge_fund['rate'])
        doge_price_change = calculate_price_change(doge_res)

        sol_rate = np.mean(sub_sol_fund['rate'])
        sol_std = np.std(sub_sol_fund['rate'])
        sol_price_change = calculate_price_change(sol_res)

        ltc_rate = np.mean(sub_ltc_fund['rate'])
        ltc_std = np.std(sub_ltc_fund['rate'])
        ltc_price_change = calculate_price_change(ltc_res)

        ada_rate = np.mean(sub_ada_fund['rate'])
        ada_std = np.std(sub_ada_fund['rate'])
        ada_price_change = calculate_price_change(ada_res)

        symbol_list = ['btc','ltc','eth','xrp','doge','sol','ada']

        for pair in itertools.combinations(symbol_list, 2):
            coin_1 = pair[0]
            coin_2 = pair[1]
            if coin_1 == 'btc':
                coin1_rate = btc_rate
                coin1_std = btc_std
                coin1_change_value = btc_price_change
            elif coin_1 == 'ltc':
                coin1_rate = ltc_rate
                coin1_std = ltc_std
                coin1_change_value = ltc_price_change
            elif coin_1 == 'eth':
                coin1_rate = eth_rate
                coin1_std = eth_std
                coin1_change_value = eth_price_change
            elif coin_1 == 'xrp':
                coin1_rate = xrp_rate
                coin1_std = xrp_std
                coin1_change_value = xrp_price_change
            elif coin_1 == 'sol':
                coin1_rate = sol_rate
                coin1_std = sol_std
                coin1_change_value = sol_price_change
            elif coin_1 == 'doge':
                coin1_rate = doge_rate
                coin1_std = doge_std
                coin1_change_value = doge_price_change
            elif coin_1 == 'ada':
                coin1_rate = ada_rate
                coin1_std = ada_std
                coin1_change_value = ada_price_change
            else:
                p = 1
            if coin_2 == 'btc':
                coin2_rate = btc_rate
                coin2_std = btc_std
                coin2_change_value = btc_price_change
            elif coin_2 == 'ltc':
                coin2_rate = ltc_rate
                coin2_std = ltc_std
                coin2_change_value = ltc_price_change
            elif coin_2 == 'eth':
                coin2_rate = eth_rate
                coin2_std = eth_std
                coin2_change_value = eth_price_change
            elif coin_2 == 'xrp':
                coin2_rate = xrp_rate
                coin2_std = xrp_std
                coin2_change_value = xrp_price_change
            elif coin_2 == 'sol':
                coin2_rate = sol_rate
                coin2_std = sol_std
                coin2_change_value = sol_price_change
            elif coin_2 == 'doge':
                coin2_rate = doge_rate
                coin2_std = doge_std
                coin2_change_value = doge_price_change
            elif coin_2 == 'ada':
                coin2_rate = ada_rate
                coin2_std = ada_std
                coin2_change_value = ada_price_change
            else:
                p = 1

            ins = pd.DataFrame({'date':date_t,'coin_1_name':coin_1,'coin_2_name':coin_2,'coin_rate':coin1_rate-coin2_rate,'coin_std':coin1_std-coin2_std,'price_change_value':coin1_change_value-coin2_change_value},index=[0])
            #print(ins)
            look_df = pd.concat([look_df,ins])
        #print('last_df')
        #print(last_df)
        i += 1
    except:
        break

import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')
look_df = look_df.dropna()
date_period = list(sorted(set(look_df['date'])))
look_df = look_df[look_df.coin_1_name != 'ada']
look_df = look_df[look_df.coin_2_name != 'ada']
res_dict = {'coin_long':None,'coin_short':None,'res':None}
res_df = pd.DataFrame()

for ele in date_period:
    print(res_dict)
    ins = look_df[look_df.date==ele]
    ins['rate_abs'] = ins['coin_rate'].apply(lambda x:np.abs(x))
    
    sub_ins = ins[ins.rate_abs==np.max(ins['rate_abs'])]
    
    #sub_ins = sub_ins[sub_ins.coin_rate==np.max(sub_ins['coin_rate'])]
    sub_ins = sub_ins.reset_index(drop=True)
    if len(sub_ins)>1:
        sub_ins = sub_ins.iloc[len(sub_ins)-1:len(sub_ins)]
        sub_ins = sub_ins.reset_index(drop=True)
    
    if sub_ins['coin_rate'][0]<0:
        coin_long = sub_ins['coin_1_name'][0]
        coin_short = sub_ins['coin_2_name'][0]
        value = sub_ins['price_change_value'][0]
    else:
        coin_long = sub_ins['coin_2_name'][0]
        coin_short = sub_ins['coin_1_name'][0]
        value = -sub_ins['price_change_value'][0] 
        
        
    pre_coin_long = res_dict['coin_long']
    pre_coin_short = res_dict['coin_short']
    pre_coin_value = res_dict['res']
    
    #if coin_long == pre_coin_long and coin_short == pre_coin_short and pre_coin_value < 0:

    if coin_long == pre_coin_long and coin_short == pre_coin_short and pre_coin_value < 0:
        ins_1 = ins[(ins.coin_1_name!=coin_long)&(ins.coin_2_name!=coin_short)]
        ins_1 = ins_1[(ins_1.coin_1_name!=coin_short)&(ins_1.coin_2_name!=coin_long)]
        #print(ins)
        sub_ins = ins_1[ins_1.rate_abs==np.max(ins_1['rate_abs'])]

        #sub_ins = sub_ins[sub_ins.coin_rate==np.max(sub_ins['coin_rate'])]
        sub_ins = sub_ins.reset_index(drop=True)


        if sub_ins['coin_rate'][0]<0:
            coin_long = sub_ins['coin_1_name'][0]
            coin_short = sub_ins['coin_2_name'][0]
            value = sub_ins['price_change_value'][0]
        else:
            coin_long = sub_ins['coin_2_name'][0]
            coin_short = sub_ins['coin_1_name'][0]
            value = -sub_ins['price_change_value'][0] 
    if value < 0:
        res_dict['coin_long'] = coin_long
        res_dict['coin_short'] = coin_short
        res_dict['res'] = value
    
    date = sub_ins['date'][0]
    df = pd.DataFrame({'date':date,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])
    res_df = pd.concat([res_df,df])
res_df = res_df.reset_index(drop=True)
table_model_3 = res_df.iloc[-30:-1]
action_values_model3 = len(table_model_3)
success_values_model3 = len(table_model_3[table_model_3.value>0])/len(table_model_3)
return_values_model3 = np.sum(table_model_3['value'])


# ====================================================================== 模型4 ==================================================================
import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
#from tqdm import tqdm
from datetime import datetime,timedelta

# ======= 正式开始执行
date_now = str(datetime.utcnow())[0:10]

# 获取数据

import importlib
import sys
import os
import urllib
import requests
import base64
import json

import time
import pandas as pd
import numpy as np
import random
import hmac
import ccxt
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')
# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df
API_URL = 'https://api.bitget.com'
API_SECRET_KEY = 'ca8d708b774782ce0fd09c78ba5c19e1e421d5fd2a78964359e6eb306cf15c67'
API_KEY = 'bg_42d96db83714abb3757250cef9ba7752'
PASSPHRASE = 'HBLww130130130'
margein_coin = 'USDT'
futures_type = 'USDT-FUTURES'
contract_num = 5

def get_timestamp():
    return int(time.time() * 1000)
def sign(message, secret_key):
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d)
def pre_hash(timestamp, method, request_path, body):
    return str(timestamp) + str.upper(method) + request_path + body
def parse_params_to_str(params):
    url = '?'
    for key, value in params.items():
        url = url + str(key) + '=' + str(value) + '&'
    return url[0:-1]
def get_header(api_key, sign, timestamp, passphrase):
    header = dict()
    header['Content-Type'] = 'application/json'
    header['ACCESS-KEY'] = api_key
    header['ACCESS-SIGN'] = sign
    header['ACCESS-TIMESTAMP'] = str(timestamp)
    header['ACCESS-PASSPHRASE'] = passphrase
    # header[LOCALE] = 'zh-CN'
    return header

def truncate(number, decimals):
    factor = 10.0 ** decimals
    return int(number * factor) / factor

def write_txt(content):
    with open(f"/home/liweiwei/duichong/test/process_result.txt", "a") as file:
        file.write(content)

def get_price(symbol):
    w2 = 0
    g2 = 0 
    while w2 == 0:
        try:
            timestamp = get_timestamp()
            response = None
            request_path = "/api/v2/mix/market/ticker"
            url = API_URL + request_path
            params = {"symbol":symbol,"productType":futures_type}
            request_path = request_path + parse_params_to_str(params)
            url = API_URL + request_path
            body = ""
            sign_cang = sign(pre_hash(timestamp, "GET", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_cang, timestamp, PASSPHRASE)
            response = requests.get(url, headers=header)
            ticker = json.loads(response.text)
            price_d = float(ticker['data'][0]['lastPr'])
            if price_d > 0:
                w2 = 1
            else:
                w2 = 0
            g2 += 1

        except:
            time.sleep(0.2)
            g2 += 1
            continue
    return price_d

def open_state(crypto_usdt,order_usdt,side,tradeSide):
    logo_b = 0
    while logo_b == 0:
        try:
            timestamp = get_timestamp()
            response = None
            clientoid = "bitget%s"%(str(int(datetime.now().timestamp())))
            #print('clientoid'+clientoid)
            request_path = "/api/v2/mix/order/place-order"
            url = API_URL + request_path
            params = {"symbol":crypto_usdt,"productType":futures_type,"marginCoin": margein_coin, "marginMode":"crossed","side":side,"tradeSide":tradeSide,"size":str(order_usdt),"orderType":"market","clientOid":clientoid}
            #print(params)
            body = json.dumps(params)
            sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
            header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
            response = requests.post(url, data=body, headers=header)
            buy_res_base = json.loads(response.text)
            #print("响应内容 (文本)---1:", buy_res_base)
            buy_id_base = int(buy_res_base['data']['orderId'])
            if buy_id_base  > 10:
                logo_b = 1
            else:
                logo_b = 0
        except:
            time.sleep(0.2)
            continue
    return buy_id_base

def check_order(crypto_usdt,id_num):
    logo_s = 0
    while logo_s == 0:
        try:
            timestamp = get_timestamp()
            response = None
            request_path_mix = "/api/v2/mix/order/detail"
            params_mix = {"symbol":crypto_usdt,"productType":futures_type,"orderId":str(id_num)}
            request_path_mix = request_path_mix + parse_params_to_str(params_mix)
            url = API_URL + request_path_mix
            body = ""
            sign_mix = sign(pre_hash(timestamp, "GET", request_path_mix,str(body)), API_SECRET_KEY)
            header_mix = get_header(API_KEY, sign_mix, timestamp, PASSPHRASE)
            response_mix = requests.get(url, headers=header_mix)

            response_1 = json.loads(response_mix.text)

            base_price = float(response_1['data']['priceAvg'])             
            base_num = float(response_1['data']['baseVolume'])
            if base_price >0 and base_num > 0:
                logo_s = 1
            else:
                logo_s = 0
        except:
            time.sleep(0.2)
            continue
    return base_price,base_num

def get_bitget_klines(symbol,endTime,granularity):
    timestamp = get_timestamp()
    response = None
    request_path_mix = "/api/v2/mix/market/history-candles"
    params_mix = {"symbol":symbol,"granularity":granularity,"productType":"USDT-FUTURES","endTime": endTime,"limit": 100}
    request_path_mix = request_path_mix + parse_params_to_str(params_mix)
    url = API_URL + request_path_mix
    body = ""
    sign_mix = sign(pre_hash(timestamp, "GET", request_path_mix,str(body)), API_SECRET_KEY)
    header_mix = get_header(API_KEY, sign_mix, timestamp, PASSPHRASE)
    response_mix = requests.get(url, headers=header_mix)
    response_1 = json.loads(response_mix.text)
    return response_1["data"]

def fetch_last_month_klines(symbol, granularity_value,number):
    """
    获取最近一个月的所有15分钟K线数据
    """
    klines = pd.DataFrame()
    # 计算一个月前的时间戳（毫秒）
    one_month_ago = int((datetime.now() - timedelta(days=50)).timestamp() * 1000)
    end_time = one_month_ago
    signal = 0
    while True:
        data = get_bitget_klines(symbol,endTime=end_time,granularity=granularity_value)
        res = pd.DataFrame()
        for i in range(len(data)):
            ins = data[i]
            date_time = ins[0]
            open_price = ins[1]
            high_price = ins[2]
            low_price = ins[3]
            close_price = ins[4]
            btc_volumn = ins[5]
            usdt_volumn = ins[6]
            # 秒级时间戳
            timestamp_seconds = int(date_time)/1000
            # 转换为正常时间
            normal_time = datetime.fromtimestamp(timestamp_seconds)
            # 格式化为字符串
            formatted_time = normal_time.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame({'date_time':date_time,'formatted_time':formatted_time,'open_price':open_price,'high_price':high_price,'low_price':low_price,'close_price':close_price,'btc_volumn':btc_volumn,'usdt_volumn':usdt_volumn},index=[0])
            res = pd.concat([res,df])
        #print(res)
        klines = pd.concat([klines,res])
        #print(klines)
        res = res.sort_values(by='date_time')
        res = res.reset_index(drop=True)
        # 更新下一个请求的开始时间为最后一条数据的时间戳
        last_time = int(res['date_time'][len(res)-1])
        #print(res['formatted_time'][0])
        #print(res['formatted_time'][len(res)-1])
        end_time = last_time + number * 1000 * 100  # 加上5分钟
        #print(end_time)

        # 如果获取的数据覆盖到当前时间，则停止循环
        
        if end_time <= int(time.time() * 1000) and signal ==0:
            continue
        elif end_time > int(time.time() * 1000) and signal ==0:
            signal = 1
            continue
        else:
            break

        # 避免频繁请求API，添加适当的延时
        time.sleep(1)

    return klines

coin_list = ['btc','sol','xrp','doge','eth']
#coin_list = ['eth']
for c_ele in coin_list:
    symbol = c_ele.upper() + 'USDT'
    print(symbol)
    data_15m_name = c_ele + '_15m_data.csv'
    #data_1m = fetch_last_month_klines(symbol,granularity_value='1m',number=60)
    data_15m = fetch_last_month_klines(symbol,granularity_value='15m',number=900)
    #data_1m.to_csv(data_1m_name)
    data_15m.to_csv(data_15m_name)
import importlib
import sys
import os
import urllib
import requests
import base64
import json

import time
import pandas as pd
import numpy as np
import random
import hmac
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
# 禁止所有警告
warnings.filterwarnings('ignore')

btc_data = pd.read_csv('btc_15m_data.csv')
sol_data = pd.read_csv('sol_15m_data.csv')
eth_data = pd.read_csv('eth_15m_data.csv')
xrp_data = pd.read_csv('xrp_15m_data.csv')
doge_data = pd.read_csv('doge_15m_data.csv')

btc_data['date_time'] = btc_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
eth_data['date_time'] = eth_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
sol_data['date_time'] = sol_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
xrp_data['date_time'] = xrp_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))
doge_data['date_time'] = doge_data['formatted_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S') - timedelta(hours=8))

btc_data = btc_data.drop('formatted_time', axis=1)
eth_data = eth_data.drop('formatted_time', axis=1)
sol_data = sol_data.drop('formatted_time', axis=1)
xrp_data = xrp_data.drop('formatted_time', axis=1)
doge_data = doge_data.drop('formatted_time', axis=1)

btc_data['date'] = btc_data['date_time'].apply(lambda x:x.date())
btc_data['hour'] = btc_data['date_time'].apply(lambda x:x.hour)
btc_data['minutes'] = btc_data['date_time'].apply(lambda x:x.minute)

eth_data['date'] = eth_data['date_time'].apply(lambda x:x.date())
eth_data['hour'] = eth_data['date_time'].apply(lambda x:x.hour)
eth_data['minutes'] = eth_data['date_time'].apply(lambda x:x.minute)

sol_data['date'] = sol_data['date_time'].apply(lambda x:x.date())
sol_data['hour'] = sol_data['date_time'].apply(lambda x:x.hour)
sol_data['minutes'] = sol_data['date_time'].apply(lambda x:x.minute)

xrp_data['date'] = xrp_data['date_time'].apply(lambda x:x.date())
xrp_data['hour'] = xrp_data['date_time'].apply(lambda x:x.hour)
xrp_data['minutes'] = xrp_data['date_time'].apply(lambda x:x.minute)

doge_data['date'] = doge_data['date_time'].apply(lambda x:x.date())
doge_data['hour'] = doge_data['date_time'].apply(lambda x:x.hour)
doge_data['minutes'] = doge_data['date_time'].apply(lambda x:x.minute)

date_list = list(sorted(set(btc_data['date'])))
date_p = str(date_list[1])
data_l = str(date_list[-2])

print(date_p,data_l)
btc_data_15m = btc_data[(btc_data.date>=pd.to_datetime(date_p))&(btc_data.date<=pd.to_datetime(data_l))]
sol_data_15m = sol_data[(sol_data.date>=pd.to_datetime(date_p))&(sol_data.date<=pd.to_datetime(data_l))]
eth_data_15m = eth_data[(eth_data.date>=pd.to_datetime(date_p))&(eth_data.date<=pd.to_datetime(data_l))]
xrp_data_15m = xrp_data[(xrp_data.date>=pd.to_datetime(date_p))&(xrp_data.date<=pd.to_datetime(data_l))]
doge_data_15m = doge_data[(doge_data.date>=pd.to_datetime(date_p))&(doge_data.date<=pd.to_datetime(data_l))]
btc_data_15m = btc_data_15m.sort_values(by='date_time')
sol_data_15m = sol_data_15m.sort_values(by='date_time')
eth_data_15m = eth_data_15m.sort_values(by='date_time')
xrp_data_15m = xrp_data_15m.sort_values(by='date_time')
doge_data_15m = doge_data_15m.sort_values(by='date_time')
print(np.max(btc_data_15m['date']),np.min(btc_data_15m['date']))
print(np.max(sol_data_15m['date']),np.min(sol_data_15m['date']))
print(np.max(eth_data_15m['date']),np.min(eth_data_15m['date']))
print(np.max(xrp_data_15m['date']),np.min(xrp_data_15m['date']))
print(np.max(doge_data_15m['date']),np.min(doge_data_15m['date']))

def calculate_bov(df):
    df = df.sort_values(by='date_time')
    df = df.reset_index(drop=True)
    # 假设 df 是 BTC 历史数据
    df['returns'] = df['close_price'].pct_change()
    df['volatility_30d'] = df['returns'].rolling(30).std()
    df['volatility_90d'] = df['returns'].rolling(90).std()
    # 归一化（0-100）
    df['volatility_score'] = (df['volatility_30d'] / df['volatility_30d'].max() * 100)
    
    bov = df['volatility_score'][len(df)-1]
    return bov
date_period = list(sorted(set(btc_data_15m['date'])))
length = len(date_period)
last_df = pd.DataFrame()
i = 0
while i < length-1:
    try:
        date_interval = date_period[i:i+4]
        date_0 = date_interval[0]
        date_1 = date_interval[2]
        date_2 = date_interval[3]
        print(date_0,date_1,date_2)
        # =======================。前一天的参数
        btc_data_15m_sample = btc_data_15m[(btc_data_15m.date>=date_0)&(btc_data_15m.date<=date_1)]
        btc_data_15m_sample = btc_data_15m_sample.sort_values(by='date_time')
        sol_data_15m_sample = sol_data_15m[(sol_data_15m.date>=date_0)&(sol_data_15m.date<=date_1)]
        sol_data_15m_sample = sol_data_15m_sample.sort_values(by='date_time')
        eth_data_15m_sample = eth_data_15m[(eth_data_15m.date>=date_0)&(eth_data_15m.date<=date_1)]
        eth_data_15m_sample = eth_data_15m_sample.sort_values(by='date_time')
        xrp_data_15m_sample = xrp_data_15m[(xrp_data_15m.date>=date_0)&(xrp_data_15m.date<=date_1)]
        xrp_data_15m_sample = xrp_data_15m_sample.sort_values(by='date_time')
        doge_data_15m_sample = doge_data_15m[(doge_data_15m.date>=date_0)&(doge_data_15m.date<=date_1)]
        doge_data_15m_sample = doge_data_15m_sample.sort_values(by='date_time')

        # ======================== 后一天的结果
        btc_data_15m_result = btc_data_15m[btc_data_15m.date==date_2]
        btc_data_15m_result = btc_data_15m_result.sort_values(by='date_time')
        btc_data_15m_result = btc_data_15m_result.reset_index(drop=True)
        sol_data_15m_result = sol_data_15m[sol_data_15m.date==date_2]
        sol_data_15m_result = sol_data_15m_result.sort_values(by='date_time')
        sol_data_15m_result = sol_data_15m_result.reset_index(drop=True)
        eth_data_15m_result = eth_data_15m[eth_data_15m.date==date_2]
        eth_data_15m_result = eth_data_15m_result.sort_values(by='date_time')
        eth_data_15m_result = eth_data_15m_result.reset_index(drop=True)
        xrp_data_15m_result = xrp_data_15m[xrp_data_15m.date==date_2]
        xrp_data_15m_result = xrp_data_15m_result.sort_values(by='date_time')
        xrp_data_15m_result = xrp_data_15m_result.reset_index(drop=True)
        doge_data_15m_result = doge_data_15m[doge_data_15m.date==date_2]
        doge_data_15m_result = doge_data_15m_result.sort_values(by='date_time')
        doge_data_15m_result = doge_data_15m_result.reset_index(drop=True)


        symbol_list = ['btc','eth','xrp','doge','sol']


        for pair in itertools.combinations(symbol_list, 2):
        #for pair in symbol_list:
            coin_1 = pair[0]
            coin_2 =  pair[1]
            if coin_1 == 'btc':
                coin1_data = btc_data_15m_sample
                coin1_bov = calculate_bov(btc_data_15m_sample)
                coin1_first_value = btc_data_15m_result['open_price'].iloc[0]
                coin1_price_change = (btc_data_15m_result['close_price'][len(btc_data_15m_result)-1]- coin1_first_value) / coin1_first_value
            elif coin_1 == 'eth':
                coin1_data = eth_data_15m_sample
                coin1_bov = calculate_bov(eth_data_15m_sample)
                coin1_first_value = eth_data_15m_result['open_price'].iloc[0]
                coin1_price_change = (eth_data_15m_result['close_price'][len(eth_data_15m_result)-1]- coin1_first_value) / coin1_first_value
            elif coin_1 == 'xrp':
                coin1_data = xrp_data_15m_sample
                coin1_bov = calculate_bov(xrp_data_15m_sample)
                coin1_first_value = xrp_data_15m_result['open_price'].iloc[0]
                coin1_price_change = (xrp_data_15m_result['close_price'][len(xrp_data_15m_result)-1]- coin1_first_value) / coin1_first_value
            elif coin_1 == 'sol':
                coin1_data = sol_data_15m_sample
                coin1_bov = calculate_bov(sol_data_15m_sample)
                coin1_first_value = sol_data_15m_result['open_price'].iloc[0]
                coin1_price_change = (sol_data_15m_result['close_price'][len(sol_data_15m_result)-1]- coin1_first_value) / coin1_first_value
            elif coin_1 == 'doge':
                coin1_data = doge_data_15m_sample
                coin1_bov = calculate_bov(doge_data_15m_sample)
                coin1_first_value = doge_data_15m_result['open_price'].iloc[0]
                coin1_price_change = (doge_data_15m_result['close_price'][len(doge_data_15m_result)-1]- coin1_first_value) / coin1_first_value
            else:
                p = 1
            if coin_2 == 'btc':
                coin2_data = btc_data_15m_sample
                coin2_bov = calculate_bov(btc_data_15m_sample)
                coin2_first_value = btc_data_15m_result['open_price'].iloc[0]
                coin2_price_change = (btc_data_15m_result['close_price'][len(btc_data_15m_result)-1]- coin2_first_value) / coin2_first_value   
            elif coin_2 == 'eth':
                coin2_data = eth_data_15m_sample
                coin2_bov = calculate_bov(eth_data_15m_sample)
                coin2_first_value = eth_data_15m_result['open_price'].iloc[0]
                coin2_price_change = (eth_data_15m_result['close_price'][len(eth_data_15m_result)-1]- coin2_first_value) / coin2_first_value
            elif coin_2 == 'xrp':
                coin2_data = xrp_data_15m_sample
                coin2_bov = calculate_bov(xrp_data_15m_sample)
                coin2_first_value = xrp_data_15m_result['open_price'].iloc[0]
                coin2_price_change = (xrp_data_15m_result['close_price'][len(xrp_data_15m_result)-1]- coin2_first_value) / coin2_first_value
            elif coin_2 == 'sol':
                coin2_data = sol_data_15m_sample
                coin2_bov = calculate_bov(sol_data_15m_sample)
                coin2_first_value = sol_data_15m_result['open_price'].iloc[0]
                coin2_price_change = (sol_data_15m_result['close_price'][len(sol_data_15m_result)-1]- coin2_first_value) / coin2_first_value
            elif coin_2 == 'doge':
                coin2_data = doge_data_15m_sample
                coin2_bov = calculate_bov(doge_data_15m_sample)
                coin2_first_value = doge_data_15m_result['open_price'].iloc[0]
                coin2_price_change = (doge_data_15m_result['close_price'][len(doge_data_15m_result)-1]- coin2_first_value) / coin2_first_value
            else:
                p = 1


            new_data = coin1_data.merge(coin2_data,how='inner',on=['date_time'])
            new_data = new_data.sort_values(by='date_time')
            new_data = new_data.reset_index(drop=True)

            # 进行协整分析

            import statsmodels.api as sm
            from statsmodels.tsa.stattools import adfuller
            price1 = new_data['close_price_x']
            price2 = new_data['close_price_y']
            #print(price1,price2)

            # 对数收益率
            log_ret1 = np.log(price1 / price1.shift(1)).dropna()
            log_ret2 = np.log(price2 / price2.shift(1)).dropna()

            # Engle-Granger 2步法
            # 1. 进行回归分析
            X = sm.add_constant(log_ret2)  # 自变量（加常数项）
            model = sm.OLS(log_ret1, X).fit()
            residuals = model.resid
            #print(residuals)

            # 2. 检验回归残差是否平稳（ADF检验）
            adf_result = adfuller(residuals)


            # 提取收盘价
            corr_value = new_data['close_price_x'].corr(new_data['close_price_y'])

            # 计算价格比例
            new_data['price_percent'] = new_data['close_price_x'] / new_data['close_price_y']

            per_mean = np.mean(new_data['price_percent'])
            per_std = np.std(new_data['price_percent'])

            deviation_degree = (new_data['price_percent'][len(new_data)-1]-per_mean)/per_std


            ins = pd.DataFrame({'date':date_1,'coin_1_name':coin_1,'coin_2_name':coin_2,'deviation_degree':deviation_degree,'corr_value':corr_value,'bov_d':coin1_bov - coin2_bov,'price_change':coin1_price_change-coin2_price_change},index=[0])
            #print(ins)
            last_df = pd.concat([last_df,ins])
        i += 1
    except:
        break
date_period = list(sorted(set(last_df['date'])))
res_dict = {'coin_long':'sol','coin_short':'ltc','res':-1}
look_df = pd.DataFrame()
for date in date_period:
    value = 0
    ins = last_df[last_df.date==date]
    ins = ins[(ins.corr_value>0.7)]
    ins = ins.reset_index(drop=True)
    if len(ins)>0:
        ins['d_abs'] = ins['deviation_degree'].apply(lambda x:np.abs(x))
        sub_ins = ins[ins.d_abs==np.max(ins['d_abs'])]
        sub_ins = sub_ins.reset_index(drop=True)
        if sub_ins['deviation_degree'][0] > 1.5:
            coin_long = sub_ins['coin_1_name'][0]
            coin_short = sub_ins['coin_2_name'][0]
            value = sub_ins['price_change'][0]
        elif sub_ins['deviation_degree'][0] < -1.5:
            coin_long = sub_ins['coin_2_name'][0]
            coin_short = sub_ins['coin_1_name'][0]
            value = -sub_ins['price_change'][0] 
        else:
            coin_long = None
            coin_short = None
            value = 0

        pre_coin_long = res_dict['coin_long']
        pre_coin_short = res_dict['coin_short']
        pre_coin_value = res_dict['res']
        if coin_long == pre_coin_long and coin_short == pre_coin_short :
            ins_1 = ins[(ins.coin_1_name!=coin_long)&(ins.coin_2_name!=coin_short)]
            ins_1 = ins_1[(ins_1.coin_1_name!=coin_short)&(ins_1.coin_2_name!=coin_long)]
            #print(ins)
            if len(ins_1)>0:
                sub_ins = ins_1[ins_1.d_abs==np.max(ins_1['d_abs'])]
                sub_ins = sub_ins.reset_index(drop=True)
                if sub_ins['deviation_degree'][0] > 1.5:
                    coin_long = sub_ins['coin_1_name'][0]
                    coin_short = sub_ins['coin_2_name'][0]
                    value = sub_ins['price_change'][0]
                elif sub_ins['deviation_degree'][0] < -1.5:
                    coin_long = sub_ins['coin_2_name'][0]
                    coin_short = sub_ins['coin_1_name'][0]
                    value = -sub_ins['price_change'][0]

                else:
                    coin_long = None
                    coin_short = None
                    value = 0
                
        if value <0:
            res_dict['coin_long'] = coin_long
            res_dict['coin_short'] = coin_short
            res_dict['res'] = value
        date = ins['date'][0]

        df = pd.DataFrame({'date':date,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])
        #df = pd.DataFrame({'date':date,'value':np.mean(sub_ins['value'])},index=[0])

        look_df = pd.concat([look_df,df])
look_df = look_df.dropna()
look_df = look_df.reset_index(drop=True)
table_model_4 = look_df.iloc[-30:-1]
action_values_model4 = len(table_model_4)
success_values_model4 = len(table_model_4[table_model_4.value>0])/len(table_model_4)
return_values_model4 = np.sum(table_model_4['value'])

#======自动发邮件
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
# 将DataFrame转换为HTML表格
html_table1 = table_model_2.to_html(index=False)
html_table2 = table_model_1.to_html(index=False)
html_table3 = table_model_3.to_html(index=False)
html_table4 = table_model_4.to_html(index=False)
# 定义HTML内容，包含两个表格
html_content = f"""
<html>
  <body>
    <p>您好，</p>
    <p>以下是第4模型表格，执行次数：{action_values_model4},成功率为：{success_values_model4},总收益为：{return_values_model4}：</p>
    {html_table4}
    <br>
    <p>以下是第3模型表格，执行次数：{action_values_model3},成功率为：{success_values_model3},总收益为：{return_values_model3}：</p>
    {html_table3}
    <br>
    <p>以下是第2模型表格，执行次数：{action_values_model2},成功率为：{success_values_model2},总收益为：{return_values_model2}：</p>
    {html_table1}
    <br>
    <p>以下是第1模型表格，执行次数：{action_values},成功率为：{success_values}：</p>
    {html_table2}
    <br>
    <p>祝好，<br>卡森</p>
  </body>
</html>
"""
#设置服务器所需信息
#163邮箱服务器地址
mail_host = 'smtp.163.com'  
#163用户名
mail_user = 'lee_daowei@163.com'  
#密码(部分邮箱为授权码) 
mail_pass = 'GKXGKVGTYBGRMAVE'   
#邮件发送方邮箱地址
sender = 'lee_daowei@163.com'  

#邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
receivers = ['lee_daowei@163.com']  
context = f'多币种对冲数据历史回顾{date_now}'
email_sender(mail_host,mail_user,mail_pass,sender,receivers,context,html_content)