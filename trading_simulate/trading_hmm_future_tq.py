# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:02 2018

@author: lion95
"""

from __future__ import division
import sys
print(sys.path)
sys.path.append('C:\\Users\\51951\\PycharmProjects\\joinquant')  # 新加入的
print(sys.path)
import os
import datetime
from jqdatasdk import *
import copy
from email_fuction import send_email
from trading_future.future_singleton import Future
# auth('18610039264', 'zg19491001')
from configDB import *
auth(JOINQUANT_USER, JOINQUANT_PW)

from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")

from tqsdk import TqApi, TqSim, TqAccount
import pandas as pd
import numpy as np
import time
import datetime
from trading_simulate.trading_fuction import Trading


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='post',
                     count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


def get_date(calen):
    EndDate = calen[-1]
    hq_last_date = calen[-2]
    return calen, EndDate, str(hq_last_date)[:10]


def cap_vol_by_rolling(vol, target_vol):
    idxs = vol.index
    for idx in range(len(idxs)):
        curDate = idxs[idx]
        vol[curDate] = max(vol[curDate], target_vol)
    return vol


def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.open = pn_data['open']
        self.high = pn_data['high']
        self.low = pn_data['low']
        self.close = pn_data['close']
        self.volume = pn_data['volume']
        # self.amount=pn_data['amount']
        # self.returns = self.close-self.close.shift(1)

    def alpha000(self):
        data_m = self.close.rolling(10).apply(values_deviation, raw=True)
        return data_m

def get_dataset(data_ori, factor_lst, future_period):
    dataset = data_ori.loc[:, ['open', 'high', 'low', 'close', 'volume']]
    for alpha in factor_lst:
        alpha = 'Alpha.' + alpha
        Alpha = Alphas(dataset)
        dataset[alpha[-8:]] = eval(alpha)()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = pd.concat(
        [dataset[factor_lst], data_ori[['high', 'low', 'close', 'volume', 'open', 'trade_date', 'close_1']]],
        axis=1)
    # 日涨跌幅，模型训练的Y值
    dataset['ret'] = dataset['close'].shift(-future_period) / dataset['close'] - 1
    dataset['ret'] = dataset['ret'].fillna(0)
    dataset = dataset.dropna()
    return dataset

def get_best_hmm_model(X, max_iter=10000, max_states=6):
    best_score = -(10 ** 10)
    best_state = 0
    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components=state, random_state=100, covariance_type='diag',
                                n_iter=max_iter).fit(X)
        try:
            if hmm_model.score(X) > best_score:
                best_score = hmm_model.score(X)
                best_state = state
        except:
            continue
    best_model = GaussianHMM(n_components=best_state, random_state=100, covariance_type='diag',
                             n_iter=max_iter).fit(X)
    return best_model

# 隐状态定义涨跌：单一状态下累计损益为正即为看涨信号，反之为看跌信号
def get_longshort_state_from_cumsum(dataset, cols_features, max_states):
    net_new = copy.deepcopy(dataset).dropna()
    X = np.column_stack([net_new.loc[:, cols_features]])
    model = get_best_hmm_model(X, max_states=max_states)
    hidden_states = model.predict(X)
    long_states = []
    short_states = []
    random_states = []
    for k in range(model.n_components):
        idx = (hidden_states == k)
        idx_int = idx.astype(int)
        net_new['%dth_hidden_state' % k] = idx_int
        net_new['%dth_ret' % k] = net_new['%dth_hidden_state' % k] * net_new['ret']
        if net_new['%dth_ret' % k].sum() > 0:
            long_states.append(k)
        elif net_new['%dth_ret' % k].sum() < 0:
            short_states.append(k)
        elif net_new['%dth_ret' % k].sum() == 0:
            random_states.append(k)
    print('做多隐状态：%s, 做空隐状态：%s, 空仓隐状态：%s' %(long_states, short_states, random_states))
    return model, hidden_states, long_states, short_states, random_states

def get_hmm_pos_df_all(index_code_lst, train_period, test_period, index_hq_dic, future_period,
                       factor_lst, max_states):
    pos_df_all = []

    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        index_hq = index_hq_dic[index_code]
        # index_hq = index_hq[index_hq['trade_date'] <= hq_last_date]
        data_set = get_dataset(index_hq, factor_lst, future_period)
        for i in range(train_period, len(data_set), test_period):
            if i + test_period >= len(data_set):
                train_set = data_set.iloc[i - train_period:i]
                test_set = data_set.iloc[i:]
            else:
                continue
        model, hidden_states, long_states, short_states, random_states = get_longshort_state_from_cumsum(
            train_set, factor_lst, max_states)

        hidden_states_predict = model.predict(test_set[factor_lst])
        date = test_set.index.tolist()[-1]
        pos = None
        predict = hidden_states_predict[-1]
        if predict in long_states:
            pos = 1
        elif predict in short_states:
            pos = -1
        elif predict in random_states:
            pos = 0
        pos_df_all.append([date, index_code, pos])
    pos_df_all = pd.DataFrame(pos_df_all, columns=['trade_date', 'symbol', 'weight'])
    pos_df_all.index = pos_df_all['symbol']
    return pos_df_all


def stock_price(sec, period, sday, eday, length):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = api.get_kline_serial(index_code, 86400, data_length=length)
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'trade_date'}).dropna()
    temp['stock_code'] = sec
    temp['trade_date'] = temp['trade_date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    temp.index = temp['trade_date']
    temp['close_1'] = temp['close'].shift(1)
    return temp


def get_signal(signal, aum, balance, EndDate, close_dict):
    symbol_lst = signal.index.tolist()
    porfolio = Future()
    main_contract_dict = porfolio.get_main_symbol(product=symbol_lst, date=EndDate)
    print(main_contract_dict)
    contract_lst = [main_contract_dict[i] for i in symbol_lst]
    ExchangeID_dict = porfolio.get_ExchangeID(contract_lst=contract_lst)
    ExchangeInstID_dict = porfolio.get_ExchangeInstID(contract_lst=contract_lst)
    VolumeMultiple_dict = porfolio.get_VolumeMultiple(contract_lst)

    signal_dict = {}
    for symbol in symbol_lst:
        main_contract = main_contract_dict[symbol]
        trading_code = ExchangeID_dict[main_contract]['ExchangeID'] + '.' + ExchangeInstID_dict[main_contract][
            'ExchangeInstID']
        signal_dict[symbol] = {
            'symbol': symbol, 'trading_code': trading_code, 'weight': signal.loc[symbol]['weight'],
            'last_price': close_dict[symbol],
            'VolumeMultiple': VolumeMultiple_dict[main_contract]['VolumeMultiple']
        }
    trading_info = pd.DataFrame(signal_dict).T
    trading_info['position'] = aum * balance / len(symbol_lst) * trading_info['weight'] / trading_info[
        'last_price'] / trading_info['VolumeMultiple']
    # trading_info['position'] = trading_info['position'].apply(lambda x: int(np.around(x, 0)))
    print(trading_info)
    return trading_info


if __name__ == '__main__':
    api = TqApi(TqAccount("simnow", "168694", "zg19491001"), web_gui=False)
    Trd = Trading(api)

    aum = 10000000
    balance = 6
    strategy_id = 'hmm'
    fold_path = 'c://g//trading_hmm//'
    # 收件人为多个收件人
    # receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
    receiver = ['xiahutao@163.com', '3467518502@qq.com', '542362275@qq.com']
    today = datetime.date.today()
    index_code_lst = ['AG', 'I', 'JM', 'MA', 'PP', 'RM', 'RU', 'SC', 'SM', 'TA', 'IF', 'Y', 'SN', 'ZC', 'AP', 'HC',
                   'AU', 'P', 'RB', 'V', 'B', 'CU', 'CF', 'L', 'IH', 'J', 'NI', 'IC', 'AL', 'BU', 'FG', 'JD', 'M', 'ZN',
                      'A', 'SF', 'OI', 'SR']  # sharp>0.2所有品种
    # index_code_lst = ['I']
    normalize_code_future = get_normal_future_index_code()
    N = 100
    num = 0
    StartDate = '2017-01-01'
    bars = 252 * 3
    train_period = 240
    test_period = 60
    future_period = 1
    factor_lst = ['alpha000']
    max_states = 6
    calen = get_trade_days(count=bars)
    calen = list(calen)
    if today in calen:
        calen, EndDate, hq_last_date = get_date(calen)
        index_hq_dic = {}
        EndDate = EndDate.strftime('%Y-%m-%d')
        date = EndDate
        close_dict = {}
        for index_code in index_code_lst:
            code = normalize_code_future[index_code]

            index_hq = stock_price(code, 'daily', StartDate, EndDate)
            index_hq_dic[index_code] = index_hq
            close_dict[index_code] = index_hq[index_hq['trade_date'] == EndDate].close.tolist()[0]
        pos_df_all_ymjh = get_hmm_pos_df_all(index_code_lst, train_period, test_period, index_hq_dic, future_period,
                       factor_lst, max_states)
        print(pos_df_all_ymjh)

        res = pos_df_all_ymjh.sort_values(by='trade_date', ascending=False)
        # res.index = range(len(res))
        res_n = res.copy()
        res_n = res_n[res_n['trade_date'] == EndDate].drop(['trade_date'], axis=1).set_index(['symbol'])
        # res_n.columns = ['weight']
        print(res_n)
        res_n.to_csv(fold_path + 'temp//' + strategy_id + '_' + EndDate + '.csv')
        # send_email(res_n, date, receiver)

        trading_info = get_signal(res_n, aum, balance, EndDate, close_dict)
        trading_info.to_csv(fold_path + 'position_hmm_' + EndDate + '.csv')
        subject = date + strategy_id
        send_email(trading_info, subject, receiver)

        trading_info['position'] = trading_info['position'].apply(lambda x: int(np.around(x, 0)))
        trading_info.index = trading_info['trading_code']
        print(trading_info)
        code_lst = trading_info.trading_code.tolist()
        while datetime.datetime.now().hour < 15:
            print('==========================================================================================')
            orders = api.get_order()
            for oid, order in orders.items():
                if order.status == 'ALIVE':
                    print(order.status)
                    api.cancel_order(order)

            positions = api.get_position()
            for symbol, order in positions.items():
                if symbol not in code_lst:
                    if order.pos_long > 0:
                        Trd.insert_order_sp_limit(symbol)
                    if order.pos_short > 0:
                        Trd.insert_order_bp_limit(symbol)

            for code in code_lst:
                position_account = api.get_position(code)
                position_long = position_account.pos_long
                position_short = position_account.pos_short
                position = trading_info.loc[code]['position']
                if code == 'DCE.y2009':
                    a = 0
                if position == 0 and position_short == 0 and position_long == 0:
                    print('%s:   仓位%s手, 状态：%s' % (code, position, '完成'))
                    continue
                elif position == position_long and position_short == 0:
                    print('%s: 多头持仓%s手, 状态：%s' % (code, position, '完成'))
                    continue
                elif position == -position_short and position_long == 0:
                    print('%s: 空头持仓%s手, 状态：%s' % (code, position, '完成'))
                    continue
                else:
                    print('%s:   仓位%s手, 状态：%s' % (code, position, '未完成'))
                quote = api.get_quote(code)
                if position > 0:
                    if position_short > 0:
                        order_bp = Trd.insert_order_bp_limit(code)
                    diff = position - position_long
                    if diff > 0:
                        order = Trd.insert_order_bk_limit(code, int(diff))
                    elif diff < 0:
                        order = Trd.insert_order_sp_limit(code, -int(diff))
                if position < 0:
                    if position_long > 0:
                        order_sp = Trd.insert_order_sp_limit(code)
                    diff = -position - position_short
                    if diff > 0:
                        order = Trd.insert_order_sk_limit(code, int(diff))
                    elif diff < 0:
                        order = Trd.insert_order_bp_limit(code, -int(diff))
                if position == 0:
                    if position_short > 0:
                        order_bp = Trd.insert_order_bp_limit(code)
                    if position_long > 0:
                        order_sp = Trd.insert_order_sp_limit(code)
            t_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[-8:]
            time.sleep(60)





