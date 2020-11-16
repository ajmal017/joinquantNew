# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:02 2018

@author: lion95
"""

from __future__ import division
import sys
print(sys.path)
sys.path.append('C:\\Users\\51951\\PycharmProjects\\joinquantNew')  # 新加入的
print(sys.path)
import os
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
# from tqsdk import TqApi, TqSim, TqAccount
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


def get_dataset_open(data_ori, factor_lst, future_period):
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
    dataset['ret'] = dataset['open'].shift(-future_period-1) / dataset['open'].shift(-1) - 1
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


def trans_state_to_bs(x, long_states, short_states, random_states):
    pos = None
    if x in long_states:
        pos = 1
    elif x in short_states:
        pos = -1
    elif x in random_states:
        pos = 0
    return pos


def get_hmm_pos_df_all(index_code_lst, param_dict, index_hq_dic, future_period,
                       factor_lst, max_states, name_lst):
    pos_df_all = pd.DataFrame([], columns=['trade_date'])
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        param = param_dict[index_code]
        index_name = name_lst[j]
        index_hq = index_hq_dic[index_code]
        data_set = get_dataset(index_hq, factor_lst, future_period)
        pos_df = data_set[['trade_date']].reset_index(drop=True)
        for m in range(len(param)):
            (train_period, test_period) = param[m]
            for i in range(train_period, len(data_set), test_period):
                if i + test_period >= len(data_set):
                    train_set = data_set.iloc[i - train_period:i]
                    test_set = data_set.iloc[i:]
                else:
                    continue
            model, hidden_states, long_states, short_states, random_states = get_longshort_state_from_cumsum(
                train_set, factor_lst, max_states)
            hidden_states_predict = []
            for n in range(len(test_set)):
                hidden_states_predict.append(model.predict(test_set[factor_lst].head(n + 1))[-1])
            # hidden_states_predict = model.predict(test_set[factor_lst])
            pos_temp = test_set[['trade_date']].reset_index(drop=True)
            pos_temp['pos%s' % m] = hidden_states_predict
            pos_temp['pos%s' % m] = pos_temp['pos%s' % m].apply(lambda x: trans_state_to_bs(x, long_states, short_states, random_states))
            pos_df = pos_df.merge(pos_temp, on=['trade_date'])
        pos_df = pos_df.set_index(['trade_date'])
        pos_df[index_name] = pos_df.mean(axis=1)

        pos_df = pos_df.reset_index(drop=False)
        pos_df_all = pos_df_all.merge(pos_df[['trade_date', index_name]], on=['trade_date'], how='outer')
    return pos_df_all


def get_hmm_pos_df_all_open(index_code_lst, param_dict, index_hq_dic, future_period,
                       factor_lst, max_states, name_lst):
    pos_df_all = pd.DataFrame([], columns=['trade_date'])
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        param = param_dict[index_code]
        index_name = name_lst[j]
        index_hq = index_hq_dic[index_code]
        data_set = get_dataset_open(index_hq, factor_lst, future_period)
        pos_df = data_set[['trade_date']].reset_index(drop=True)
        for m in range(len(param)):
            (train_period, test_period) = param[m]
            for i in range(train_period, len(data_set), test_period):
                if i + test_period >= len(data_set):
                    train_set = data_set.iloc[i - train_period:i]
                    test_set = data_set.iloc[i:]
                else:
                    continue
            model, hidden_states, long_states, short_states, random_states = get_longshort_state_from_cumsum(
                train_set, factor_lst, max_states)
            hidden_states_predict = []
            for n in range(len(test_set)):
                hidden_states_predict.append(model.predict(test_set[factor_lst].head(n+1))[-1])
            # hidden_states_predict = model.predict(test_set[factor_lst])
            pos_temp = test_set[['trade_date']].reset_index(drop=True)
            pos_temp['pos%s' % m] = hidden_states_predict
            pos_temp['pos%s' % m] = pos_temp['pos%s' % m].apply(
                lambda x: trans_state_to_bs(x, long_states, short_states, random_states))
            pos_df = pos_df.merge(pos_temp, on=['trade_date'])
        pos_df = pos_df.set_index(['trade_date'])
        pos_df['o' + index_name] = pos_df.mean(axis=1)
        pos_df = pos_df.reset_index(drop=False)
        pos_df_all = pos_df_all.merge(pos_df[['trade_date', 'o' + index_name]], on=['trade_date'], how='outer')
    return pos_df_all


def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
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
    # api = TqApi(TqAccount("simnow", "176793", "yo193846"), web_gui=False)
    # api = TqApi()
    # Trd = Trading(api)
    aum = 10000000
    balance = 6
    strategy_id = 'hmm'
    fold_path = 'c://e//hmm//resualt//stockindex//'
    # 收件人为多个收件人
    # receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
    receiver = ['xiahutao@163.com', '3467518502@qq.com', '542362275@qq.com']
    today = datetime.date.today()
    asset_lst = ['000300.XSHG', '000016.XSHG', '000905.XSHG', '399006.XSHE']
    code_name_lst = ['沪深300', '上证50', '中证500', '创业板']
    param_all = pd.read_csv(fold_path + 'para_opt_history.csv') \
        .assign(s_date=lambda df: df.s_date.apply(lambda x: str(x))) \
        .assign(e_date=lambda df: df.e_date.apply(lambda x: str(x)))
    param_dict = {i: {} for i in asset_lst}
    for i in range(len(asset_lst)):
        code = asset_lst[i]
        temp = param_all[param_all['asset'] == code]
        # print(len(temp))
        train_days_lst = temp['train_days'].tolist()
        test_days_lst = temp['test_days'].tolist()
        param_dict[code] = [
            (train_days_lst[i], test_days_lst[i]) for i in range(len(train_days_lst))]
    N = 100
    num = 0
    StartDate = '2017-01-01'
    bars = 252 * 3
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
        for index_code in asset_lst:
            code = index_code
            index_hq = stock_price(code, 'daily', StartDate, EndDate)
            index_hq_dic[index_code] = index_hq
            close_dict[index_code] = index_hq[index_hq['trade_date'] == EndDate].close.tolist()[0]
        pos_df_all_ymjh = get_hmm_pos_df_all(asset_lst, param_dict, index_hq_dic, future_period,
                       factor_lst, max_states, code_name_lst)
        print(pos_df_all_ymjh)

        pos_df_all_ymjh_open = get_hmm_pos_df_all_open(asset_lst, param_dict, index_hq_dic, future_period,
                                             factor_lst, max_states, code_name_lst)
        print(pos_df_all_ymjh_open)
        ret = pos_df_all_ymjh.merge(pos_df_all_ymjh_open, on=['trade_date'])

        res = ret.sort_values(by='trade_date', ascending=False)
        res.index = range(len(res))
        res_n = res.copy()
        print(res_n)
        res_n.to_csv(fold_path + 'index' + '_' + EndDate + '.csv', encoding='gbk')
        send_email(res_n, date+'HMM_stockindex', receiver)






