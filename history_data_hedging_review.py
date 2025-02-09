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


# ================================================================================== 反向验证 =========================================================


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

test_result.columns = test_result.iloc[0]

# 删除第一行
test_result = test_result[1:].reset_index(drop=True)
#======自动发邮件
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
# 将DataFrame转换为HTML表格
html_table1 = test_result.to_html(index=False)
html_table2 = new_result.to_html(index=False)
html_table3 = look_df.to_html(index=False)

# 定义HTML内容，包含两个表格
html_content = f"""
<html>
  <body>
    <p>您好，</p>
    <p>以下是第一个表格：</p>
    {html_table1}
    <br>
    <p>以下是第二个表格，在cci和roc方向相同下，检查roc：</p>
    {html_table2}
    <br>
    <p>以下是第二个表格，执行次数：{action_values},成功率为：{success_values}：</p>
    {html_table3}
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