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
import pickle
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

def calculate_price_change(df):
    df = df.sort_values(by='date_time')
    df = df.reset_index(drop=True)
    first_value = df['open_price'][0]
    last_value = df['close_price'][len(df)-1]
    price_change = (last_value-first_value)/first_value
    return price_change

API_URL = 'https://api.bitget.com'

margein_coin = 'USDT'
futures_type = 'USDT-FUTURES'
contract_num = 20

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
    with open(f"/root/many_crypto_hedging/process_3_result.txt", "a") as file:
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
    one_month_ago = int((datetime.now() - timedelta(days=10)).timestamp() * 1000)
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

res_dict = {'coin_long':'sol','coin_short':'ltc','res':-1}
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
        import requests
        import time
        from datetime import datetime
        # 转换时间戳（秒转换为毫秒）
        def to_milliseconds(timestamp):
            return int(timestamp * 1000)

        # 获取当前时间的 Unix 时间戳（毫秒）
        def current_timestamp():
            return int(time.time() * 1000)

        # 获取 3 年前的时间戳（毫秒）
        def get_three_years_ago_timestamp():
            three_years_in_seconds = 4 * 24 * 60 * 60  # 3年 = 3 * 365 * 24 * 60 * 60 秒
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
                end_time = start_time + 86400000  # 每次请求1天的数据（86400000 毫秒 = 1天）
                
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
            funding_rates = get_funding_rates_for_three_years(symbol)
            # 打印部分结果
            for entry in funding_rates:  # 打印前10条数据
                ins = pd.DataFrame({'symbol':entry['symbol'], 'rate':float(entry['fundingRate']), 'time': entry['fundingTime']},index=[0])
                last_df = pd.concat([last_df,ins])
        last_df['date_time'] = last_df['time'].apply(lambda x: datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
        last_df['date'] = last_df['date_time'].apply(lambda x:pd.to_datetime(x).date())
        date_period = list(sorted(set(last_df['date'])))
        date_period = date_period[-4:-1]
        date_0 = date_period[0]
        date_1 = date_period[2]
        last_df = last_df[(last_df.date>=date_0)&(last_df.date<=date_1)]
        import itertools
        sub_btc_fund = last_df[last_df.symbol=='BTCUSDT']
        sub_eth_fund = last_df[last_df.symbol=='ETHUSDT']
        sub_xrp_fund = last_df[last_df.symbol=='XRPUSDT']
        sub_doge_fund = last_df[last_df.symbol=='DOGEUSDT']
        sub_sol_fund = last_df[last_df.symbol=='SOLUSDT']
        sub_ltc_fund = last_df[last_df.symbol=='LTCUSDT']
        sub_ada_fund = last_df[last_df.symbol=='ADAUSDT']


        btc_rate = np.mean(sub_btc_fund['rate'])
        btc_std = np.std(sub_btc_fund['rate'])

        eth_rate = np.mean(sub_eth_fund['rate'])
        eth_std = np.std(sub_eth_fund['rate'])

        xrp_rate = np.mean(sub_xrp_fund['rate'])
        xrp_std = np.std(sub_xrp_fund['rate'])

        doge_rate = np.mean(sub_doge_fund['rate'])
        doge_std = np.std(sub_doge_fund['rate'])

        sol_rate = np.mean(sub_sol_fund['rate'])
        sol_std = np.std(sub_sol_fund['rate'])

        ltc_rate = np.mean(sub_ltc_fund['rate'])
        ltc_std = np.std(sub_ltc_fund['rate'])

        ada_rate = np.mean(sub_ada_fund['rate'])
        ada_std = np.std(sub_ada_fund['rate'])

        symbol_list = ['btc','ltc','eth','xrp','doge','sol','ada']
        look_df = pd.DataFrame()
        for pair in itertools.combinations(symbol_list, 2):
            coin_1 = pair[0]
            coin_2 = pair[1]
            if coin_1 == 'btc':
                coin1_rate = btc_rate
                coin1_std = btc_std
            elif coin_1 == 'ltc':
                coin1_rate = ltc_rate
                coin1_std = ltc_std
            elif coin_1 == 'eth':
                coin1_rate = eth_rate
                coin1_std = eth_std
            elif coin_1 == 'xrp':
                coin1_rate = xrp_rate
                coin1_std = xrp_std
            elif coin_1 == 'sol':
                coin1_rate = sol_rate
                coin1_std = sol_std
            elif coin_1 == 'doge':
                coin1_rate = doge_rate
                coin1_std = doge_std
            elif coin_1 == 'ada':
                coin1_rate = ada_rate
                coin1_std = ada_std
            else:
                p = 1
            if coin_2 == 'btc':
                coin2_rate = btc_rate
                coin2_std = btc_std
            elif coin_2 == 'ltc':
                coin2_rate = ltc_rate
                coin2_std = ltc_std
            elif coin_2 == 'eth':
                coin2_rate = eth_rate
                coin2_std = eth_std
            elif coin_2 == 'xrp':
                coin2_rate = xrp_rate
                coin2_std = xrp_std
            elif coin_2 == 'sol':
                coin2_rate = sol_rate
                coin2_std = sol_std
            elif coin_2 == 'doge':
                coin2_rate = doge_rate
                coin2_std = doge_std
            elif coin_2 == 'ada':
                coin2_rate = ada_rate
                coin2_std = ada_std
            else:
                p = 1

            ins = pd.DataFrame({'coin_1_name':coin_1,'coin_2_name':coin_2,'coin_rate':coin1_rate-coin2_rate,'coin_std':coin1_std-coin2_std},index=[0])
            #print(ins)
            look_df = pd.concat([look_df,ins])
            
        look_df = look_df[look_df.coin_1_name!='ada']
        look_df = look_df[look_df.coin_2_name!='ada']
        look_df['rate_abs'] = look_df['coin_rate'].apply(lambda x:np.abs(x))
        sub_ins = look_df[look_df.rate_abs==np.max(look_df['rate_abs'])]
        sub_ins = sub_ins.reset_index(drop=True)

        if len(sub_ins)>1:
            sub_ins = sub_ins.iloc[len(sub_ins)-1:len(sub_ins)]
            sub_ins = sub_ins.reset_index(drop=True)

        if sub_ins['coin_rate'][0]<0:
            coin_long = sub_ins['coin_1_name'][0]
            coin_short = sub_ins['coin_2_name'][0]
        else:
            coin_long = sub_ins['coin_2_name'][0]
            coin_short = sub_ins['coin_1_name'][0]


        pre_coin_long = res_dict['coin_long']
        pre_coin_short = res_dict['coin_short']
        pre_coin_value = res_dict['res']

        if coin_long == pre_coin_long and coin_short == pre_coin_short and pre_coin_value < 0:
            ins_1 = look_df[(look_df.coin_1_name!=coin_long)&(look_df.coin_2_name!=coin_short)]
            ins_1 = ins_1[(ins_1.coin_1_name!=coin_short)&(ins_1.coin_2_name!=coin_long)]
            #print(ins)
            sub_ins = ins_1[ins_1.rate_abs==np.max(ins_1['rate_abs'])]
            sub_ins = sub_ins.reset_index(drop=True)
            if sub_ins['coin_rate'][0]<0:
                coin_long = sub_ins['coin_1_name'][0]
                coin_short = sub_ins['coin_2_name'][0]
            else:
                coin_long = sub_ins['coin_2_name'][0]
                coin_short = sub_ins['coin_1_name'][0] 

        data_target = date_period[-1]
        content_judge = f'根据{data_target}的数据判断{dt.date()}做多币种{coin_long},做空币种{coin_short}' + '\n'
        write_txt(content_judge)
        
        judge = 0
        while judge == 0:
            try:
                pairs = ['BTCUSDT', 'ETHUSDT','XRPUSDT','DOGEUSDT','SOLUSDT','LTCUSDT','ADAUSDT']
                all_volumePlace = {'BTCUSDT':0, 'ETHUSDT':0,'XRPUSDT':0,'DOGEUSDT':0,'SOLUSDT':0,'LTCUSDT':0,'ADAUSDT':0}
                all_pricePlace = {'BTCUSDT':0, 'ETHUSDT':0,'XRPUSDT':0,'DOGEUSDT':0,'SOLUSDT':0,'LTCUSDT':0,'ADAUSDT':0}
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

        coin_long_usdt = coin_long.upper()+'USDT'
        coin_short_usdt = coin_short.upper()+'USDT'
        coin_long_volumePlace = all_volumePlace[coin_long_usdt]
        coin_short_volumePlace = all_volumePlace[coin_short_usdt]

        positions = {'position': 'None','coin_long_name':coin_long_usdt,'coin_short_name':coin_short_usdt,'coin_long_num':0,'coin_long_price':0,'coin_long_fee':0,'coin_short_num':0,'coin_short_price':0,'coin_short_fee':0,'close_signal':0,'coin_long_volumePlace':coin_long_volumePlace,'coin_short_volumePlace':coin_short_volumePlace}

        order_value = 2000
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
                time.sleep(3)
                raw_coin_long_price = positions['coin_long_price']
                raw_coin_short_price = positions['coin_short_price']
                last_coin_long_price = get_price(symbol=coin_long_name)
                last_coin_short_price = get_price(symbol=coin_short_name)

                long_price_change = (last_coin_long_price-raw_coin_long_price)/raw_coin_long_price
                short_price_change = (last_coin_short_price-raw_coin_short_price)/raw_coin_short_price

                if long_price_change - short_price_change > 0.25 or long_price_change - short_price_change < -0.045:
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
                    if long_price_change - short_price_change < 0:
                        res_dict['coin_long'] = coin_long
                        res_dict['coin_short'] = coin_short
                        res_dict['res'] = long_price_change - short_price_change
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
                        if long_price_change - short_price_change < 0:
                            res_dict['coin_long'] = coin_long
                            res_dict['coin_short'] = coin_short
                            res_dict['res'] = long_price_change - short_price_change
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
