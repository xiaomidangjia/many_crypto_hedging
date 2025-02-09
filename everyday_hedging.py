# 获取近3天的数据

import importlib
import sys
import os
import urllib
import requests
import base64
import json
from datetime import datetime,timedelta
import time
import pandas as pd
import numpy as np
import random
import pandas as pd
import pandas_ta as ta
import itertools
import warnings
import hmac
# 禁止所有警告
warnings.filterwarnings('ignore')
# 计算价格变化率（Rate of Change）
def calculate_roc(df, column='close_price', window=14):
    df['ROC'] = df[column].pct_change(periods=window)  # 返回百分比变化
    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    return df
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
API_URL = 'https://api.bitget.com'

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
    with open(f"/root/many_crypto_hedging/process_result.txt", "a") as file:
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
    one_month_ago = int((datetime.now() - timedelta(days=3)).timestamp() * 1000)
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

while True:
    time.sleep(1)
    raw_process_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # 将字符串解析为 datetime 对象
    dt = datetime.fromisoformat(raw_process_time)
    # 提取日期部分,+1 表示程序要平仓的时间
    date_part = dt.date() + timedelta(days=1)
    hour_part = dt.hour
    if int(hour_part)==8:
        content_process_start = f'程序开始运行的日期{dt.date()}和小时{hour_part}'+ '\n'
        content_process_close = f'程序最终结束的日期{date_part}和小时{8}'+ '\n'
        write_txt(content_process_start)
        write_txt(content_process_close)
        coin_list = ['btc','sol','xrp','doge','eth']
        for c_ele in coin_list:
            symbol = c_ele.upper() + 'USDT'
            data_15m_name = c_ele + '_15m_data_3.csv'
            data_15m = fetch_last_month_klines(symbol,granularity_value='15m',number=900)
            data_15m.to_csv(data_15m_name)
        # 读取15分钟数据
        btc_data_15m = pd.read_csv('btc_15m_data_3.csv')
        sol_data_15m = pd.read_csv('sol_15m_data_3.csv')
        eth_data_15m = pd.read_csv('eth_15m_data_3.csv')
        xrp_data_15m = pd.read_csv('xrp_15m_data_3.csv')
        doge_data_15m = pd.read_csv('doge_15m_data_3.csv')
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


        btc_data_15m = btc_data_15m[['date_time','date','close_price','high_price','low_price']]
        sol_data_15m = sol_data_15m[['date_time','date','close_price','high_price','low_price']]
        eth_data_15m = eth_data_15m[['date_time','date','close_price','high_price','low_price']]
        xrp_data_15m = xrp_data_15m[['date_time','date','close_price','high_price','low_price']]
        doge_data_15m = doge_data_15m[['date_time','date','close_price','high_price','low_price']]


        date_list = list(sorted(set(btc_data_15m['date'])))
        data_target = str(date_list[-2])

        #print(data_target)
        btc_data_15m = btc_data_15m[btc_data_15m.date==pd.to_datetime(data_target)]
        sol_data_15m = sol_data_15m[sol_data_15m.date==pd.to_datetime(data_target)]
        eth_data_15m = eth_data_15m[eth_data_15m.date==pd.to_datetime(data_target)]
        xrp_data_15m = xrp_data_15m[xrp_data_15m.date==pd.to_datetime(data_target)]
        doge_data_15m = doge_data_15m[doge_data_15m.date==pd.to_datetime(data_target)]

        btc_data_15m = btc_data_15m.sort_values(by='date_time')
        sol_data_15m = sol_data_15m.sort_values(by='date_time')
        eth_data_15m = eth_data_15m.sort_values(by='date_time')
        xrp_data_15m = xrp_data_15m.sort_values(by='date_time')
        doge_data_15m = doge_data_15m.sort_values(by='date_time')

        # 计算 rsi
        btc_rsi = calculate_rsi(btc_data_15m)
        btc_rsi = btc_rsi.dropna()
        btc_rsi_value = np.mean(btc_rsi['RSI'])

        sol_rsi = calculate_rsi(sol_data_15m)
        sol_rsi = sol_rsi.dropna()
        sol_rsi_value = np.mean(sol_rsi['RSI'])

        eth_rsi = calculate_rsi(eth_data_15m)
        eth_rsi = eth_rsi.dropna()
        eth_rsi_value = np.mean(eth_rsi['RSI'])

        xrp_rsi = calculate_rsi(xrp_data_15m)
        xrp_rsi = xrp_rsi.dropna()
        xrp_rsi_value = np.mean(xrp_rsi['RSI'])

        doge_rsi = calculate_rsi(doge_data_15m)
        doge_rsi = doge_rsi.dropna()
        doge_rsi_value = np.mean(doge_rsi['RSI'])

        # 计算 roc
        btc_roc = calculate_roc(btc_data_15m)
        btc_roc = btc_roc.dropna()
        btc_roc_value = np.mean(btc_roc['ROC'])

        sol_roc = calculate_roc(sol_data_15m)
        sol_roc = sol_roc.dropna()
        sol_roc_value = np.mean(sol_roc['ROC'])

        eth_roc = calculate_roc(eth_data_15m)
        eth_roc = eth_roc.dropna()
        eth_roc_value = np.mean(eth_roc['ROC'])

        xrp_roc = calculate_roc(xrp_data_15m)
        xrp_roc = xrp_roc.dropna()
        xrp_roc_value = np.mean(xrp_roc['ROC'])

        doge_roc = calculate_roc(doge_data_15m)
        doge_roc = doge_roc.dropna()
        doge_roc_value = np.mean(doge_roc['ROC'])

        btc_cci = calculate_cci(btc_data_15m, n=20)
        btc_cci = btc_cci.dropna()
        btc_cci_value = np.mean(btc_cci)

        sol_cci = calculate_cci(sol_data_15m, n=20)
        sol_cci = sol_cci.dropna()
        sol_cci_value = np.mean(sol_cci)

        eth_cci = calculate_cci(eth_data_15m, n=20)
        eth_cci = eth_cci.dropna()
        eth_cci_value = np.mean(eth_cci)

        xrp_cci = calculate_cci(xrp_data_15m, n=20)
        xrp_cci = xrp_cci.dropna()
        xrp_cci_value = np.mean(xrp_cci)

        doge_cci = calculate_cci(doge_data_15m, n=20)
        doge_cci = doge_cci.dropna()
        doge_cci_value = np.mean(doge_cci)
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
        content_judge = f'根据{data_target}的数据判断{dt.date()}做多币种{coin_long},做空币种{coin_short}' + '\n'
        write_txt(content_judge)
        
        judge = 0
        while judge == 0:
            try:
                pairs = ['BTCUSDT', 'ETHUSDT','XRPUSDT','DOGEUSDT','SOLUSDT']
                all_volumePlace = {'BTCUSDT':0, 'ETHUSDT':0,'XRPUSDT':0,'DOGEUSDT':0,'SOLUSDT':0}
                all_pricePlace = {'BTCUSDT':0, 'ETHUSDT':0,'XRPUSDT':0,'DOGEUSDT':0,'SOLUSDT':0}
                for crypto_usdt in pairs:
                    # 初始化 bitget 平台的合约的 保证金模式，杠杆大小，以及开仓币种的最小下单单位
                    # 调整保证金模式（全仓/逐仓）
                    timestamp = get_timestamp()
                    response = None
                    request_path = "/api/v2/mix/account/set-margin-mode"
                    url = API_URL + request_path
                    params = {"symbol":crypto_usdt,"marginCoin":margein_coin,"productType":futures_type,"marginMode": "crossed"}
                    body = json.dumps(params)
                    sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
                    header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
                    response = requests.post(url, data=body, headers=header)
                    response_1 = json.loads(response.text)
                    response_1_res = response_1['data']['marginMode']
                    
                    content_mode = f'{crypto_usdt}调整保证金模式:'+str(response_1_res) + '\n'
                    write_txt(content_mode)

                    # 调整杠杆（全仓）
                    timestamp = get_timestamp()
                    response = None
                    request_path = "/api/v2/mix/account/set-leverage"
                    url = API_URL + request_path
                    params = {"symbol":crypto_usdt,"marginCoin":margein_coin,"productType":futures_type,"leverage": str(contract_num)}
                    body = json.dumps(params)
                    sign_tranfer = sign(pre_hash(timestamp, "POST", request_path, str(body)), API_SECRET_KEY)
                    header = get_header(API_KEY, sign_tranfer, timestamp, PASSPHRASE)
                    response = requests.post(url, data=body, headers=header)
                    response_2 = json.loads(response.text)
                    response_2_long = response_2['data']['longLeverage']
                    response_2_short = response_2['data']['shortLeverage']
                    content_leverage = f'{crypto_usdt}调整全仓杠杆:' +'多'+str(response_2_long)+'空'+str(response_2_short) + '\n'
                    write_txt(content_mode)

                    # 获取币种的价格小数位，开仓量小数位
                    timestamp = get_timestamp()
                    response = None
                    request_path = "/api/v2/mix/market/contracts"
                    url = API_URL + request_path
                    params = {"symbol":crypto_usdt,'productType':futures_type}
                    request_path = request_path + parse_params_to_str(params)
                    url = API_URL + request_path
                    body = ""
                    sign_cang = sign(pre_hash(timestamp, "GET", request_path, str(body)), API_SECRET_KEY)
                    header = get_header(API_KEY, sign_cang, timestamp, PASSPHRASE)
                    response = requests.get(url, headers=header)
                    ticker = json.loads(response.text)
                    volumePlace = int(ticker['data'][0].get('volumePlace'))
                    pricePlace = int(ticker['data'][0].get('pricePlace'))
                    content_contracts = f'{crypto_usdt}数量和价格精度：'+str(volumePlace)+str('----')+str(pricePlace) + '\n'
                    write_txt(content_contracts)

                    all_volumePlace[str(crypto_usdt)] = volumePlace
                    all_pricePlace[str(crypto_usdt)] = pricePlace
                    
                    judge = 1
            except:
                time.sleep(0.5)

        coin_long = coin_long.upper()+'USDT'
        coin_short = coin_short.upper()+'USDT'
        coin_long_volumePlace = all_volumePlace[coin_long]
        coin_short_volumePlace = all_volumePlace[coin_short]

        positions = {'position': 'None','coin_long_name':coin_long,'coin_short_name':coin_short,'coin_long_num':0,'coin_long_price':0,'coin_long_fee':0,'coin_short_num':0,'coin_short_price':0,'coin_short_fee':0,'close_signal':0,'coin_long_volumePlace':coin_long_volumePlace,'coin_short_volumePlace':coin_short_volumePlace}

        order_value = 100
        position = positions['position']
        while position in ('run_ing','None'):
            coin_long_name = positions['coin_long_name']
            coin_short_name = positions['coin_short_name']
            if position == 'None':
                coin_long_volumePlace = positions['coin_long_volumePlace']
                coin_short_volumePlace = positions['coin_short_volumePlace']

                coin_long_price = get_price(symbol=coin_long_name)
                coin_long_num = truncate(order_value*contract_num/coin_long_price, decimals=coin_long_volumePlace)
                coin_long_order_id = open_state(crypto_usdt=coin_long_name,order_usdt=coin_long_num,side='buy',tradeSide='open')
                # 获取 long 订单详情
                coin_long_price_t,coin_long_num = check_order(crypto_usdt=coin_long_name ,id_num=coin_long_order_id)
                positions['coin_long_num'] = coin_long_num
                positions['coin_long_price'] = coin_long_price_t
                # positions
                coin_short_price = get_price(symbol=coin_short_name)
                coin_short_num = truncate(order_value*contract_num/coin_short_price, decimals=coin_short_volumePlace)
                coin_short_order_id = open_state(crypto_usdt=coin_short_name,order_usdt=coin_short_num,side='sell',tradeSide='open') 
                # 获取 short 订单详情
                coin_short_price_t,coin_short_num = check_order(crypto_usdt=coin_short_name ,id_num=coin_short_order_id)
                positions['coin_short_num'] = coin_short_num
                positions['coin_short_price'] = coin_short_price_t
                # 更新 position
                positions['position'] = 'run_ing'
                current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                content_1 = f'{coin_long_name}:{coin_short_name}交易对开仓,时间:{current_time}' + '\n'
                write_txt(content_1)
                #print(positions)
                
                position = positions['position']
            elif position == 'run_ing':
                time.sleep(60)
                raw_coin_long_price = positions['coin_long_price']
                raw_coin_short_price = positions['coin_short_price']
                last_coin_long_price = get_price(symbol=coin_long_name)
                last_coin_short_price = get_price(symbol=coin_short_name)

                long_price_change = (last_coin_long_price-raw_coin_long_price)/raw_coin_long_price
                short_price_change = (last_coin_short_price-raw_coin_short_price)/raw_coin_short_price

                if long_price_change - short_price_change > 0.015 or long_price_change - short_price_change < -0.015:
                    # 平仓
                    coin_long_num = positions['coin_long_num']
                    coin_short_num = positions['coin_short_num']
                    close_long_id = open_state(crypto_usdt=coin_long_name ,order_usdt=coin_long_num,side='buy',tradeSide='close')
                    close_short_id = open_state(crypto_usdt=coin_short_name ,order_usdt=coin_short_num,side='sell',tradeSide='close')
                    positions['position'] = 'close'
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    content_2 = f'{coin_long_name}:{coin_short_name}交易对平仓,时间:{current_time}' + '\n'
                    write_txt(content_2)
                    
                    position = positions['position']
                else:
                    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    now_dt = datetime.fromisoformat(now_time)
                    # 提取日期部分
                    now_date_part = now_dt.date()
                    now_hour = now_dt.hour
                    now_minute = now_dt.minute
                    
                    # 时间到了 utc0的 00 时间
                    if now_date_part == date_part and int(now_hour)==8:
                    #if int(now_hour)==8:
                        # 平仓
                        coin_long_num = positions['coin_long_num']
                        coin_short_num = positions['coin_short_num']
                        close_long_id = open_state(crypto_usdt=coin_long_name ,order_usdt=coin_long_num,side='buy',tradeSide='close')
                        close_short_id = open_state(crypto_usdt=coin_short_name ,order_usdt=coin_short_num,side='sell',tradeSide='close')
                        positions['position'] = 'close'
                        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                        content_3 = f'{coin_long_name}:{coin_short_name}交易对平仓,时间:{current_time}' + '\n'
                        write_txt(content_3)
                        position = positions['position']
                    else:
                        position = positions['position']
                        if now_minute in (15,30,45):
                            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                            v_value = str(round((long_price_change - short_price_change)*100,4))+'%'
                            content_4 = f'{coin_long_name}:{coin_short_name}交易对在时间:{current_time}正在监控中，差距为{v_value}' + '\n'
                            write_txt(content_4) 
                        continue
            else:
                break
    else:
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        now_dt = datetime.fromisoformat(now_time)
        # 提取日期部分
        now_date_part = now_dt.date()
        now_hour = now_dt.hour
        now_minute = now_dt.minute
        if now_minute in (15,30,45):
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            content_5 = f'已经止盈或止损，程序时间监控中待重启,目前时间为：{now_time}' + '\n'
            write_txt(content_5)
