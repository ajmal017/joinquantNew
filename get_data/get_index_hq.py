# coding=utf-8
from __future__ import division
import pandas as pd
import datetime
from trading_future.future_singleton import Future
from jqdatasdk import *
from configDB import *
auth(JOINQUANT_USER, JOINQUANT_PW)


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour >= 15:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    # code_lst = temp.symbol.tolist()
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


def get_normal_future_maincontact_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '9999']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    # code_lst = temp.symbol.tolist()
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


def future_price(sec, sday, eday, fred):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """

    temp = get_price(sec, start_date=sday, end_date=eday, frequency=fred, fields=None, skip_paused=True, fq='pre',
                     count=None)[['open', 'high', 'low', 'close', 'volume', 'money']]

    if fred == 'daily':
        temp['date_time'] = temp.index
        temp['date_time'] = temp['date_time'].apply(lambda x: str(x) + str(' 00:00'))
        temp = temp.set_index(['date_time'])
    return temp


def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    # temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
    #                  skip_paused=True, fq='pre', count=None).reset_index() \
    #     .rename(columns={'index': 'trade_date'})\
    #     .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:16]))[
    #     ['trade_date', 'open', 'high', 'low', 'close', 'volume']].dropna().set_index(['trade_date'])
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     skip_paused=True, fq='pre', count=None).reset_index()\
        .rename(columns={'index': 'date_time'})\
        .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:16]))
    temp = temp[
        ['date_time', 'open', 'high', 'low', 'close', 'volume', 'money']].dropna()
    # temp['stock_code'] = sec
    return temp


if __name__ == '__main__':
    # code_dic = get_normal_future_index_code()  # 获取指数代码
    code_dic = get_normal_future_index_code()  # 获取主力合约代码
    symbol_lst = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM', 'FG', 'IF',
                   'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']


    # symbol_lst = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']
    # symbol_lst = ['SA']
    # symbol_lst = ['000300.XSHG', '000016.XSHG', '000905.XSHG', '399006.XSHE']
    date = datetime.date.today()

    sday = '2010-01-01'
    eday = '2020-12-01'
    for symbol in symbol_lst:
        code = code_dic[symbol]
        # code = symbol
        for fred in ['daily']:
            # temp = future_price(code, sday, eday, fred)
            temp = stock_price(code, fred, sday, eday)
            print(temp)
            # temp.to_csv('e:/data/future_index/' + code + '_' + fred + '.txt')
            temp.to_csv('c:/e/data/future_index/' + symbol + '_' + fred + '_index.csv')




