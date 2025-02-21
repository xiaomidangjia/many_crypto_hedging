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

import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
#from tqdm import tqdm
from datetime import datetime,timedelta

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
    one_month_ago = int((datetime.now() - timedelta(days=360*3)).timestamp() * 1000)
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
for c_ele in coin_list:
    print(c_ele)
    symbol = c_ele.upper() + 'USDT'
    data_15m_name = c_ele + '_15m_data.csv'
    data_15m = fetch_last_month_klines(symbol,granularity_value='15m',number=900)
    data_15m.to_csv(data_15m_name)
# ===============================================================================================
#
#
# ==========================================模型1
#
#
# ===============================================================================================

# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
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

# 读取15分钟数据
btc_data_15m = pd.read_csv('btc_15m_data.csv')
sol_data_15m = pd.read_csv('sol_15m_data.csv')
eth_data_15m = pd.read_csv('eth_15m_data.csv')
xrp_data_15m = pd.read_csv('xrp_15m_data.csv')
doge_data_15m = pd.read_csv('doge_15m_data.csv')

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

btc_data_15m = btc_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
sol_data_15m = sol_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
eth_data_15m = eth_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
xrp_data_15m = xrp_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]
doge_data_15m = doge_data_15m[['date_time','date','close_price','high_price','low_price','open_price']]


date_list = list(sorted(set(btc_data_15m['date'])))
date_p = str(date_list[1])
data_l = str(date_list[-2])

print(date_p,data_l)
btc_data_15m = btc_data_15m[(btc_data_15m.date>=pd.to_datetime(date_p))&(btc_data_15m.date<=pd.to_datetime(data_l))]
sol_data_15m = sol_data_15m[(sol_data_15m.date>=pd.to_datetime(date_p))&(sol_data_15m.date<=pd.to_datetime(data_l))]
eth_data_15m = eth_data_15m[(eth_data_15m.date>=pd.to_datetime(date_p))&(eth_data_15m.date<=pd.to_datetime(data_l))]
xrp_data_15m = xrp_data_15m[(xrp_data_15m.date>=pd.to_datetime(date_p))&(xrp_data_15m.date<=pd.to_datetime(data_l))]
doge_data_15m = doge_data_15m[(doge_data_15m.date>=pd.to_datetime(date_p))&(doge_data_15m.date<=pd.to_datetime(data_l))]


date_period = list(sorted(set(btc_data_15m['date'])))
length = len(date_period)

last_df = pd.DataFrame()
i = 0
while i < length-1:
    date_interval = date_period[i:i+2]
    date_0 = date_interval[0]
    date_1 = date_interval[1]
    print(date_0,date_1)
    
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

    sub_btc_data_15m_test = btc_data_15m[btc_data_15m.date==date_1]
    sub_btc_data_15m_test = sub_btc_data_15m_test.sort_values(by='date_time')
    sub_sol_data_15m_test = sol_data_15m[sol_data_15m.date==date_1]
    sub_sol_data_15m_test = sub_sol_data_15m_test.sort_values(by='date_time')
    sub_eth_data_15m_test = eth_data_15m[eth_data_15m.date==date_1]
    sub_eth_data_15m_test = sub_eth_data_15m_test.sort_values(by='date_time')
    sub_xrp_data_15m_test = xrp_data_15m[xrp_data_15m.date==date_1]
    sub_xrp_data_15m_test = sub_xrp_data_15m_test.sort_values(by='date_time')
    sub_doge_data_15m_test = doge_data_15m[doge_data_15m.date==date_1]
    sub_doge_data_15m_test = sub_doge_data_15m_test.sort_values(by='date_time')
    
    
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
    
    
    btc_price = calculate_price_change(sub_btc_data_15m_test)
    eth_price = calculate_price_change(sub_eth_data_15m_test)
    sol_price = calculate_price_change(sub_sol_data_15m_test)
    xrp_price = calculate_price_change(sub_xrp_data_15m_test)
    doge_price = calculate_price_change(sub_doge_data_15m_test)
    
    #print('roc——value')
    #print(btc_roc_value,sol_roc_value,eth_roc_value,xrp_roc_value,doge_roc_value)
    symbol_list = ['eth','xrp','doge','sol']
    

    for pair in itertools.combinations(symbol_list, 2):
        coin_1 = pair[0]
        coin_2 = pair[1]
        if coin_1 == 'btc':
            coin1_rsi_value = btc_rsi_value
            coin1_roc_value = btc_roc_value
            coin1_cci_value = btc_cci_value
            coin1_price = btc_price
        elif coin_1 == 'eth':
            coin1_rsi_value = eth_rsi_value
            coin1_roc_value = eth_roc_value
            coin1_cci_value = eth_cci_value
            coin1_price = eth_price
        elif coin_1 == 'xrp':
            coin1_rsi_value = xrp_rsi_value
            coin1_roc_value = xrp_roc_value
            coin1_cci_value = xrp_cci_value
            coin1_price = xrp_price
        elif coin_1 == 'sol':
            coin1_rsi_value = sol_rsi_value
            coin1_roc_value = sol_roc_value
            coin1_cci_value = sol_cci_value
            coin1_price = sol_price
        elif coin_1 == 'doge':
            coin1_rsi_value = doge_rsi_value
            coin1_roc_value = doge_roc_value
            coin1_cci_value = doge_cci_value
            coin1_price = doge_price
        else:
            coin1_rsi_value = 0
            coin1_roc_value = 0
        if coin_2 == 'btc':
            coin2_rsi_value = btc_rsi_value
            coin2_roc_value = btc_roc_value
            coin2_cci_value = btc_cci_value
            coin2_price = btc_price
        elif coin_2 == 'eth':
            coin2_rsi_value = eth_rsi_value
            coin2_roc_value = eth_roc_value
            coin2_cci_value = eth_cci_value
            coin2_price = eth_price
        elif coin_2 == 'xrp':
            coin2_rsi_value = xrp_rsi_value
            coin2_roc_value = xrp_roc_value
            coin2_cci_value = xrp_cci_value
            coin2_price = xrp_price
        elif coin_2 == 'sol':
            coin2_rsi_value = sol_rsi_value
            coin2_roc_value = sol_roc_value
            coin2_cci_value = sol_cci_value
            coin2_price = sol_price
        elif coin_2 == 'doge':
            coin2_rsi_value = doge_rsi_value
            coin2_roc_value = doge_roc_value
            coin2_cci_value = doge_cci_value
            coin2_price = doge_price
        else:
            coin2_rsi_value = 0
            coin2_roc_value = 0
            
        ins = pd.DataFrame({'date':date_1,'coin_1_name':coin_1,'coin_2_name':coin_2,'coin1_rsi_value':coin1_rsi_value,'coin2_rsi_value':coin2_rsi_value,'rsi_d_abs':(coin1_rsi_value-coin2_rsi_value)/np.abs(coin1_rsi_value-coin2_rsi_value),'coin1_roc_value':coin1_roc_value,'coin2_roc_value':coin2_roc_value,'roc_d':coin1_roc_value-coin2_roc_value,'roc_d_abs':(coin1_roc_value-coin2_roc_value)/np.abs(coin1_roc_value-coin2_roc_value),'cci_d':coin1_cci_value-coin2_cci_value,'cci_d_abs':(coin1_cci_value-coin2_cci_value)/np.abs(coin1_cci_value-coin2_cci_value),'price_change':coin1_price-coin2_price},index=[0])
        last_df = pd.concat([last_df,ins])
    #print('last_df')
    #print(last_df)
    i += 1


table_m1 = pd.DataFrame()

for ele in list(sorted(set(last_df['date']))):
    print(ele)

    sub_last_df = last_df[last_df.date==ele]

    last_df_1 = sub_last_df[(sub_last_df.cci_d_abs==1) &(sub_last_df.roc_d_abs==1)]
    last_df_2 = sub_last_df[(sub_last_df.cci_d_abs==-1)&(sub_last_df.roc_d_abs==-1)]
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
            value = sub_last_df_1['price_change'][0] 
        else:
            # 选2 最小
            last_df_2['flag'] = last_df_2['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_2 else 0)
            sub_last_df_2 = last_df_2[last_df_2.flag==1]
            sub_last_df_2 = sub_last_df_2.reset_index(drop=True)
            
            # 做多coin2，做空coin1
            coin_long = sub_last_df_2['coin_2_name'][0] 
            coin_short = sub_last_df_2['coin_1_name'][0]  
            value = -sub_last_df_2['price_change'][0] 
            
    elif len(last_df_1) > 0 and len(last_df_2)==0:
        max_abs_value_1 = last_df_1['roc_d'].abs().max()

        last_df_1['flag'] = last_df_1['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_1 else 0) 
        sub_last_df_1 = last_df_1[last_df_1.flag==1]
        sub_last_df_1 = sub_last_df_1.reset_index(drop=True)

        # 做多coin1，做空coin2
        coin_long = sub_last_df_1['coin_1_name'][0] 
        coin_short = sub_last_df_1['coin_2_name'][0] 
        value = sub_last_df_1['price_change'][0] 

    elif len(last_df_1) == 0 and len(last_df_2)>0:
        max_abs_value_2 = last_df_2['roc_d'].abs().max()
 
        last_df_2['flag'] = last_df_2['roc_d'].apply(lambda x:1 if np.abs(x)==max_abs_value_2 else 0)
        sub_last_df_2 = last_df_2[last_df_2.flag==1]
        sub_last_df_2 = sub_last_df_2.reset_index(drop=True)

        # 做多coin2，做空coin1
        coin_long = sub_last_df_2['coin_2_name'][0] 
        coin_short = sub_last_df_2['coin_1_name'][0] 
        value = -sub_last_df_2['price_change'][0]
    else:
        coin_long = None
        coin_short = None


    open_df = pd.DataFrame({'date':ele,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])
    table_m1 = pd.concat([table_m1,open_df])
    
# ===============================================================================================
#
#
# ==========================================模型2
#
#
# ===============================================================================================
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
from datetime import datetime,timedelta
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

date_period = list(sorted(set(last_df['date'])))
table_m2 = pd.DataFrame()
for date in date_period:#[pd.to_datetime('2025-01-20'),pd.to_datetime('2025-01-22')]:
    ins = last_df[last_df.date==date]
    X_test = ins[features]

    with open('random_forest_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    # 预测均值
    y_pred = loaded_model.predict(X_test)
    ins['y_pred'] = y_pred
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
    table_m2 = pd.concat([table_m2,df])

# ===============================================================================================
#
#
# ==========================================模型3
#
#
# ===============================================================================================
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
print(date_period)
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

table_m3 = pd.DataFrame()

for ele in date_period:
    #print(res_dict)

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
        

    
    date = sub_ins['date'][0]
    df = pd.DataFrame({'date':date,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])
    table_m3 = pd.concat([table_m3,df])
# ===============================================================================================
#
#
# ==========================================模型4
#
#
# ===============================================================================================
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
from datetime import datetime,timedelta
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


            ins = pd.DataFrame({'date':date_2,'coin_1_name':coin_1,'coin_2_name':coin_2,'deviation_degree':deviation_degree,'corr_value':corr_value,'bov_d':coin1_bov - coin2_bov,'price_change':coin1_price_change-coin2_price_change},index=[0])
            #print(ins)
            last_df = pd.concat([last_df,ins])
        i += 1
    except:
        break
date_period = list(sorted(set(last_df['date'])))
table_m4 = pd.DataFrame()
for date in date_period:

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
    else:
        coin_long = None
        coin_short = None
        value = 0

    df = pd.DataFrame({'date':date,'coin_long':coin_long,'coin_short':coin_short,'value':value},index=[0])
    #df = pd.DataFrame({'date':date,'value':np.mean(sub_ins['value'])},index=[0])

    table_m4 = pd.concat([table_m4,df])
table_m1.rename(columns={'coin_long':'long_m1','coin_short':'short_m1','value':'m1_value'},inplace=True)
table_m2.rename(columns={'coin_long':'long_m2','coin_short':'short_m2','value':'m2_value'},inplace=True)
table_m3.rename(columns={'coin_long':'long_m3','coin_short':'short_m3','value':'m3_value'},inplace=True)
table_m4.rename(columns={'coin_long':'long_m4','coin_short':'short_m4','value':'m4_value'},inplace=True)
table_m1['date'] = table_m1['date'].apply(lambda x:pd.to_datetime(x))
table_m2['date'] = table_m2['date'].apply(lambda x:pd.to_datetime(x))
table_m3['date'] = table_m3['date'].apply(lambda x:pd.to_datetime(x))
table_m4['date'] = table_m4['date'].apply(lambda x:pd.to_datetime(x))
last_df = table_m1[['date','long_m1','short_m1','m1_value']].merge(table_m2[['date','long_m2','short_m2','m2_value']],how='left',on=['date']).merge(table_m3[['date','long_m3','short_m3','m3_value']],how='left',on=['date']).merge(table_m4[['date','long_m4','short_m4','m4_value']],how='left',on=['date'])
last_df = last_df[last_df.date>=pd.to_datetime('2023-01-01')]
last_df = last_df.fillna(0)
last_df = last_df.sort_values(by='date')
last_df = last_df.reset_index(drop=True)
sub_last_df_last = last_df.iloc[-30:]
last_df['mean_value'] = 0.3*last_df['m1_value']+0.25*last_df['m2_value']+0.25*last_df['m3_value']+0.25*last_df['m4_value']
last_df['month'] = last_df['date'].apply(lambda x:x.month)
last_df['year'] = last_df['date'].apply(lambda x:x.year)
all_df_1 = last_df.groupby(['year','month'],as_index=False)['mean_value'].sum()
all_df_2 = last_df.groupby(['year'],as_index=False)['mean_value'].sum()
all_df = pd.concat([all_df_1,all_df_2])
table_m1 = table_m1.reset_index(drop=True)
sub_table_m1 = table_m1.iloc[-30:]
table_m2 = table_m2.reset_index(drop=True)
sub_table_m2 = table_m2.iloc[-30:]
table_m3 = table_m3.reset_index(drop=True)
sub_table_m3 = table_m3.iloc[-30:]
table_m4 = table_m4.reset_index(drop=True)
sub_table_m4 = table_m4.iloc[-30:]
number = len(sub_table_m1)
m1_success = round(len(sub_table_m1[sub_table_m1.m1_value>0])/number,4)
m1_return = round(np.sum(sub_table_m1['m1_value']),4)

m2_success = round(len(sub_table_m2[sub_table_m2.m2_value>0])/number,4)
m2_return = round(np.sum(sub_table_m2['m2_value']),4)

m3_success = round(len(sub_table_m3[sub_table_m3.m3_value>0])/number,4)
m3_return = round(np.sum(sub_table_m3['m3_value']),4)

m4_success = round(len(sub_table_m4[sub_table_m4.m4_value>0])/len(sub_table_m4[sub_table_m4.m4_value!=0]),4)
m4_return = round(np.sum(sub_table_m4['m4_value']),4)


#======自动发邮件
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
# 将DataFrame转换为HTML表格
html_table1 = sub_last_df_last.to_html(index=False)
html_table2 = all_df.to_html(index=False)
# 定义HTML内容，包含两个表格
html_content = f"""
<html>
  <body>
    <p>您好，</p>
    <p>以下是明细表：</p>
    {html_table1}
    <br>
    <p>以下总表：</p>
    {html_table2}
    <br>
    <p>总执行天数：{number}：</p>
    <p>模型1--成功率：{m1_success},总收益为：{m1_return}：</p>
    <p>模型2--成功率：{m2_success},总收益为：{m2_return}：</p>
    <p>模型3--成功率：{m3_success},总收益为：{m3_return}：</p>
    <p>模型4--成功率：{m4_success},总收益为：{m4_return}：</p>

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