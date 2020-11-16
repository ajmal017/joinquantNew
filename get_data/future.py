# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/13 11:35
@Author  : zhangfang
@File    : future.py
'''
from jqdatasdk import *
from configDB import *
auth(JOINQUANT_USER, JOINQUANT_PW)
import warnings
import time
import datetime


def transfer_exchangeid(x):
    x = x.split('.')[1]
    if x == 'XSGE':
        return 'SHFE'
    elif x == 'XDCE':
        return 'DCE'
    elif x == 'XZCE':
        return 'CZCE'
    elif x == 'CCFX':
        return 'CFFEX'
    elif x == 'XINE':
        return 'INE'



def get_normal_future_index_code(code_lst, date):
    temp_all = get_all_securities(types=['futures'])
    temp_all['jq_code'] = temp_all.index
    temp_all = temp_all.reset_index(drop=True)
    temp_all['idx'] = temp_all['jq_code'].apply(lambda x: x[-9:-5])
    temp = temp_all[temp_all['idx'] == '8888'][['jq_code']].rename(columns={'jq_code': 'index_code'})
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    temp = temp[temp['symbol'].isin(code_lst)]
    temp['ExchangeID'] = temp['index_code'].apply(lambda x: transfer_exchangeid(x))
    temp['jq_code'] = temp['symbol'].apply(lambda x: get_dominant_future(x, date))

    temp = temp.merge(temp_all[['jq_code', 'name']], on=['jq_code'])
    temp = temp.rename(columns={'name': 'ExchangeInstID', 'jq_code': 'main_contract'})
    temp['tradeId'] = temp['ExchangeID'] + '.' + temp['ExchangeInstID']

    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


if __name__ == '__main__':
    # 获取所有期货列表

    asset_lst = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'FG',
                 'HC', 'I', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                 'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',
                 'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']
    a = get_normal_future_index_code(asset_lst, datetime.date.today().strftime('%Y-%m-%d'))
    b = 0
    df = get_price('C2101.XDCE', end_date='2020-11-13 12:00:00', count=5, frequency='1m')
    print(df)