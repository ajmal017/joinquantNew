# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:15
# @Author  : zhangfang
import pandas as pd
import html5lib
from trading_future.future_singleton import Future
from backtest_func import yearsharpRatio, maxRetrace, annROR, annROR_signal
import matplotlib.pyplot as plt
import numpy as np


def PowerSetsRecursive(items):
    # 求集合的所有子集
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


if __name__ == "__main__":
    fold_ini_path = 'e://Strategy//MT4//'
    # fold_ini_path = 'G://缠论//回测报告//'
    level = 1
    date_lst = [('2020-01-01', '2020-07-01'), ('2015-01-01', '2017-01-01'), ('2017-01-01', '2020-01-01')]
    # date_lst = [('2017-01-01', '2020-07-01')]

    fee = np.float(0.00015)
    porfolio = Future()
    period_ini_lst = [15, 30, 60, 240, 1440]
    period_ini_lst = [5, 15, 30, 60, 240, 1440]
    period_lst_all = PowerSetsRecursive(period_ini_lst)
    period_lst_all = [i for i in period_lst_all if len(i) == 1]
    # period_lst_all = [[5, 15, 30, 60, 240, 1440]]

    print(period_lst_all)

    code_lst = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
                'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有品种32个
    # code_lst_5 = ['ap', 'j', 'i', 'fu', 'sm', 'if', 'v', 'zn', 'pp', 'ni', 'pb']  # 所有5分钟夏普>0
    # code_lst_15 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'sc', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'hc', 'au', 'sm', 'sr', 'ta']  # 所有15分钟夏普>0
    # code_lst_30 = ['ap', 'al', 'fu', 'i', 'j', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn',
    #             'if', 'sf', 'sm', 'ta']  # 所有30分钟夏普>0
    # code_lst_60 = ['ap', 'al', 'cu', 'fu', 'i', 'j', 'ni', 'rb', 'sc', 'tf', 'v', 'zc', 'zn',
    #             'hc', 'sm', 'bu', 'ta', 'ma']  # 所有60分钟夏普>0
    # code_lst_240 = ['ap', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'pp', 'rb', 'v', 'zc', 'zn', 'c',
    #             'if', 'p', 'hc', 'jm', 'sm', 'bu', 'ta', 'ma']  # 所有4小时夏普>0
    # code_lst_1440 = ['ag', 'cu', 'j', 'pp', 'tf', 'v', 'zn', 'c',
    #               'au', 'jm', 'sm', 'bu', 'ta', 'ma']  # 所有日级别夏普>0

    # code_lst_5 = []  # 所有5分钟夏普>0
    code_lst_15 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'sc', 'v', 'zc', 'zn', 'c',
                   'if', 'sf', 'hc', 'au', 'sm', 'sr', 'ta']  # 所有15分钟夏普>0
    code_lst_30 = ['ap', 'al', 'fu', 'i', 'j', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn',
                   'if', 'sf', 'sm', 'ta']  # 所有30分钟夏普>0
    code_lst_60 = ['ap', 'al', 'cu', 'fu', 'i', 'j', 'ni', 'rb', 'sc', 'tf', 'v', 'zc', 'zn',
                   'hc', 'sm', 'bu', 'ta', 'ma']  # 所有60分钟夏普>0
    code_lst_240 = ['ap', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'pp', 'rb', 'v', 'zc', 'zn', 'c',
                    'if', 'p', 'hc', 'jm', 'sm', 'bu', 'ta', 'ma']  # 所有4小时夏普>0
    code_lst_1440 = ['ag', 'cu', 'j', 'pp', 'tf', 'v', 'zn', 'c',
                     'au', 'jm', 'sm', 'bu', 'ta', 'ma']  # 所有日级别夏普>0
    code_lst_5 = ['ap', 'j', 'i', 'sf', 'pp', 'sm', 'fu', 'if', 'zn', 'sc', 'ag', 'p', 'pb']  # 所有5分钟夏普>0

    code_dict = {}
    code_dict['5'] = code_lst_5
    code_dict['15'] = code_lst_15
    code_dict['30'] = code_lst_30
    code_dict['60'] = code_lst_60
    code_dict['240'] = code_lst_240
    code_dict['1440'] = code_lst_1440

    porfolio_lst = []
    signal_lst = []
    for (s_date, e_date) in date_lst:
        for period_lst in period_lst_all:
            print(period_lst)
            pos_df_all = pd.DataFrame(columns=['date_time'])
            pos_name_lst = []
            period_num_name_dict = {}
            for code in code_lst:
                hq = pd.read_csv('e:/data/future_index/' + code.upper() + '_' + 'daily' + '_index.csv')[
                    ['date_time', 'close']].assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))

                hq = hq[(hq['date_time'] > s_date) & (hq['date_time'] < e_date)]
                if len(hq) < 20:
                    continue
                print(code)
                profit_lst = []
                period_name = []
                for period in period_lst:
                    if code not in code_dict[str(period)]:
                        continue
                    period_name.append(period)
                    mode = '蓝线笔_蓝线反转确认_蓝线反转平仓_200627_' + str(period) + '分钟_12tick_0710笔'
                    fold_path = fold_ini_path + mode + '//'
                    html = pd.read_html(fold_path + code + '.htm', encoding='gbk')
                    state = html[0]
                    html = pd.read_html(fold_path + code + '.htm', encoding='gbk', header=0, index_col=0)
                    trade = html[1]
                    trade = trade.rename(columns={'时间': 'date_time', '获利': 'profit'}).fillna(value=0).sort_values(['date_time'])
                    trade['date_time'] = trade['date_time'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])
                    # trade = trade.groupby(['date_time'])
                    profit_df = hq.merge(trade, on=['date_time'], how='outer')
                    # trade_df = trade.sum()
                    pos_lst = []
                    pos = 0
                    order = 0
                    for date_time, group in profit_df.groupby(['date_time']):
                        len_open = len(group[(group['类型'] == 'sell') | (group['类型'] == 'buy')])
                        len_close = len(group[(group['类型'] == 's/l') | (group['类型'] == 't/p')
                                              | (group['类型'] == 'close')])

                        pos_diff = len_open - len_close
                        if len_open + len_close >= 2:
                            pos_lst.append(2)
                        elif len_open + len_close == 1:
                            pos_lst.append(max(pos, 1))
                        else:
                            pos_lst.append(pos)
                        pos = pos + pos_diff
                    pos_df = profit_df[['date_time']].groupby(['date_time']).count()
                    # pos_df['date_time'] = pos_df.index
                    pos_name = 'pos_' + code + '_' + str(period)
                    pos_df[pos_name] = pos_lst
                    pos_name_lst.append(pos_name)
                    pos_df_all = pos_df_all.merge(pos_df, on=['date_time'], how='outer')

            pos_df_all = pos_df_all.fillna(value=0)
            pos_df = pos_df_all.sort_values(['date_time']).set_index(['date_time'])
            pos_df['pos'] = pos_df[pos_name_lst].mean(axis=1) / 2
            print(pos_df)
            pos_df.to_csv(fold_ini_path + 'state_blue_line//pos_mean_' + '_'.join([str(i) for i in period_lst]) + '.csv', encoding='gbk')




