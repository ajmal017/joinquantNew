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
    date_lst = [('2015-01-01', '2020-07-01')]
    signal_date_lst = [('2015-01-01', '2016-01-01'), ('2016-01-01', '2017-01-01'), ('2017-01-01', '2018-01-01'),
                       ('2018-01-01', '2019-01-01'), ('2019-01-01', '2020-01-01'), ('2020-01-01', '2020-07-01')]
    method = 'sum'
    fee = np.float(0.00015)
    porfolio = Future()
    period_ini_lst = [15, 30, 60, 240, 1440]
    period_ini_lst = [5, 15, 30, 60, 240, 1440]
    period_lst_all = PowerSetsRecursive(period_ini_lst)
    period_lst_all = [i for i in period_lst_all if len(i) == 1]
    # period_lst_all = [[5, 15, 30, 60, 240, 1440]]
    # period_lst_all = [[15, 30, 60, 240]]
    print(period_lst_all)
    #  一筛133
    code_lst = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
                'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有品种32个
    code_lst_5 = ['sm', 'j', 'zc', 'ap', 'al', 'c', 'i', 'pp', 'sc', 'ag', 'bu', 'hc', 'ma', 'ni', 'p', 'rb', 'sf', 'ta', 'v']  # 所有夏普>0年份胜率>=0.5
    code_lst_15 = ['hc', 'j', 'sr', 'al', 'ma', 'ni', 'sm', 'ap', 'cf', 'fu', 'i', 'if', 'jm',
                   'pb', 'pp', 'sc', 'tf', 'v', 'zn', 'ag', 'bu', 'c', 'cu', 'p', 'ta', 'zc']  # 所有夏普>0年份胜率>=0.5
    code_lst_30 = ['fu', 'hc', 'v', 'al', 'ap', 'cu', 'sc', 'ta', 'tf', 'cf', 'i', 'if', 'j', 'jm', 'ma', 'p', 'pp', 'ru', 'sf', 'sr', 'zc', 'zn']  # 所有夏普>0年份胜率>=0.5
    code_lst_60 = ['ap', 'bu', 'j', 'rb', 'al', 'hc', 'pb', 'ta', 'v', 'zn', 'cu', 'm', 'ma', 'ni', 'oi', 'pp', 'ru', 'sr', 'tf']  # 所有夏普>0年份胜率>=0.5
    code_lst_240 = ['c', 'fu', 'cu', 'i', 'p', 'ta', 'v', 'zn', 'zc', 'al', 'ap', 'j', 'm', 'pp',
                    'tf', 'ag', 'hc', 'if', 'jm', 'ma', 'ni', 'pb', 'rb', 'ru', 'sm', 'sr']  # 所有夏普>0年份胜率>=0.5
    code_lst_1440 = ['fu', 'ma', 'pp', 'sm', 'tf', 'ag', 'p', 'v', 'au',
                     'al', 'bu', 'cu', 'j', 'pb', 'rb', 'sf', 'c', 'ni', 'oi', 'zc', 'zn']  # 所有夏普>0年份胜率>=0.5
    #  二筛78
    # code_lst_5 = ['sm', 'j', 'zc', 'ap', 'al', 'c', 'i', 'pp', 'sc']  # 所有夏普>0年份胜率>0.5
    # code_lst_15 = ['hc', 'j', 'sr', 'al', 'ma', 'ni', 'sm', 'ap', 'cf', 'fu', 'i', 'if', 'jm',
    #                'pb', 'pp', 'sc', 'tf', 'v', 'zn']  # 所有夏普>0年份胜率>0.5
    # code_lst_30 = ['fu', 'hc', 'v', 'al', 'ap', 'cu', 'sc', 'ta', 'tf', ]  # 所有夏普>0年份胜率>0.5
    # code_lst_60 = ['ap', 'bu', 'j', 'rb', 'al', 'hc', 'pb', 'ta', 'v', 'zn', ]  # 所有夏普>0年份胜率>0.5
    # code_lst_240 = ['c', 'fu', 'cu', 'i', 'p', 'ta', 'v', 'zn', 'zc', 'al', 'ap', 'j', 'm', 'pp',
    #                 'tf',]  # 所有夏普>0年份胜率>0.5
    # code_lst_1440 = ['fu', 'ma', 'pp', 'sm', 'tf', 'ag', 'p', 'v', 'au',
    #               'al', 'bu', 'cu', 'j', 'pb', 'rb', 'sf']  # 所有夏普>0年份胜率>0.5

    # code_lst_5 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有5分钟夏普>0
    # code_lst_15 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有15分钟夏普>0
    # code_lst_30 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有30分钟夏普>0
    # code_lst_60 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有60分钟夏普>0
    # code_lst_240 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有4小时夏普>0
    # code_lst_1440 = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
    #             'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有日级别夏普>0
    code_lst_5 = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']
    code_lst_15 = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']
    code_lst_30 = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']
    code_lst_60 = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']
    code_lst_240 = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']
    code_lst_1440 = ['A', 'B', 'CS', 'CJ', 'FG', 'L', 'JD', 'SA']

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
            chg_df_all = pd.DataFrame(columns=['date_time'])
            chg_name = []
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
                    mode = '缠论003_200724_' + str(period) + '分钟'
                    fold_path = fold_ini_path + mode + '//'
                    html = pd.read_html(fold_path + code + '.htm', encoding='gbk')
                    state = html[0]
                    html = pd.read_html(fold_path + code + '.htm', encoding='gbk', header=0, index_col=0)
                    trade = html[1]
                    profit_lst.append(trade[['时间', '获利']])
                if len(period_name) == 0:
                    continue
                profit_df_all = pd.concat(profit_lst).rename(columns={'时间': 'date_time', '获利': 'profit'}).fillna(value=0)
                profit_df_all['date_time'] = profit_df_all['date_time'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:10])
                profit_df_all = profit_df_all.groupby(['date_time'])
                profit_df = profit_df_all.sum()
                profit_df['count'] = profit_df_all.count()
                # trade_times_everyday = count_df.profit.mean()
                profit_df['date_time'] = profit_df.index
                profit_df = profit_df.assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
                profit_df = profit_df.reset_index(drop=True)
                contract_lst = [code.upper()]
                VolumeMultiple = porfolio.get_VolumeMultiple(contract_lst)[code.upper()]['VolumeMultiple']

                # aum_ini = hq.close.tolist()[0] * VolumeMultiple * 2 * level
                profit_df = hq.merge(profit_df, on=['date_time'], how='left').sort_values(['date_time'])
                # profit_df = profit_df.fillna(0)
                profit_df['chg'] = (profit_df['profit'] - profit_df['close'].shift(1) * profit_df['count'] * VolumeMultiple * fee) * level / \
                                   (profit_df['close'].shift(1) * VolumeMultiple * 2 * len(period_name))
                profit_df = profit_df.fillna(0)
                profit_df['net'] = 1 + profit_df['chg'].cumsum()
                for (sdate, edate) in signal_date_lst:
                    profit_df_ = profit_df[(profit_df['date_time'] > sdate) & (profit_df['date_time'] < edate)]
                    # print(profit_df)
                    if len(profit_df_) == 0:
                        continue
                    net_lst = profit_df_.net.tolist()
                    try:
                        sharpe_ratio = yearsharpRatio(net_lst, 1)
                    except:
                        sharpe_ratio = -1
                    try:
                        ann_return = annROR(net_lst, 1)
                    except:
                        ann_return = -1
                    try:
                        max_drawdown = maxRetrace(net_lst, 1)
                    except:
                        max_drawdown = 1

                    signal_row = []
                    signal_row.append(code)
                    signal_row.append('_'.join([str(i) for i in period_name]))
                    signal_row.append(len(period_name))
                    signal_row.append(fee)
                    signal_row.append(sharpe_ratio)
                    signal_row.append(ann_return)
                    signal_row.append(max_drawdown)
                    signal_row.append(sdate)
                    signal_row.append(edate)
                    signal_row.append(method)
                    signal_lst.append(signal_row)
                    title_str = '%s 周期%sm sharp %.2f annRet %.2f maxRetrace %.2f' % (
                        code, '_'.join([str(i) for i in period_name]), sharpe_ratio, 100 * ann_return, 100 * max_drawdown)
                    # profit_df.set_index(['date_time']).ix[:, ['net']].plot()
                    # plt.rcParams['font.sans-serif'] = ['SimHei']
                    # plt.title(title_str)
                    # plt.savefig(fold_ini_path + 'fig/' + code + '_' + '_'.join([str(i) for i in period_name]) + '.png')
                    # plt.show()
                chg_df_ = profit_df.reset_index(drop=False)[['date_time', 'chg']].rename(columns={'chg': 'chg_' + code})
                # chg_df_['period_num_' + code] = len(period_name)
                chg_df_['chg_' + code] = chg_df_['chg_' + code] * len(period_name)
                chg_name.append('chg_' + code)
                period_num_name_dict[code] = len(period_name)
                # period_num_name.append('period_num_' + code)
                chg_df_all = chg_df_all.merge(chg_df_, on=['date_time'], how='outer')
            chg_df_all = chg_df_all.fillna(value=0)
            chg_df = chg_df_all.sort_values(['date_time']).set_index(['date_time'])
            # chg_name = ['chg_' + code for code in code_lst]
            period_sum = sum([period_num_name_dict[code[4:]] for code in chg_name])
            chg_df['chg'] = chg_df[chg_name].sum(axis=1) / period_sum
            chg_df = chg_df.fillna(value=0)
            if method == 'sum':
                chg_df['net'] = 1 + chg_df['chg'].cumsum()
            else:
                chg_df['net'] = (1 + chg_df['chg']).cumprod()
            chg_df = chg_df.reset_index(drop=False)
            chg_df['date_time'] = pd.to_datetime(chg_df['date_time'])
            chg_df = chg_df.set_index(['date_time'])
            sharpe_ratio = yearsharpRatio(chg_df['net'].tolist(), 1)
            ann_return = annROR(chg_df['net'].tolist(), 1)
            max_drawdown = maxRetrace(chg_df['net'].tolist(), 1)
            porfolio_row = []
            porfolio_row.append(len(code_dict[str(period)]))
            porfolio_row.append('_'.join([str(i) for i in period_lst]))
            porfolio_row.append(fee)
            porfolio_row.append(sharpe_ratio)
            porfolio_row.append(ann_return)
            porfolio_row.append(max_drawdown)
            porfolio_row.append(s_date)
            porfolio_row.append(e_date)
            porfolio_row.append(method)
            porfolio_lst.append(porfolio_row)

            chg_df.ix[:, ['net']].plot()
            title_str = '周期%sm sharp %.2f annRet %.2f maxRetrace %.2f' % (
                '_'.join([str(i) for i in period_lst]), sharpe_ratio, 100 * ann_return,
                100 * max_drawdown)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.title(title_str)
            plt.savefig(fold_ini_path + 'fig/' + str(len(code_lst)) + '_' + '_'.join([str(i) for i in period_lst]) + 'm' + '.png')
            plt.show()

    porfolio_state = pd.DataFrame(porfolio_lst, columns=['品种数', 'period', 'fee', 'sharpe_ratio', 'ann_return',
                                                         'max_drawdown', 's_date', 'e_date', 'method'])
    porfolio_state.to_excel(fold_ini_path + 'state_blue_line//state_porfolio_003_' + '0803_50.xlsx', encoding='gbk')

    signal_state = pd.DataFrame(signal_lst, columns=['品种', 'period', 'period_num', 'fee', 'sharpe_ratio', 'ann_return',
                                                         'max_drawdown', 's_date', 'e_date', 'method'])
    signal_state.to_excel(
        fold_ini_path + 'state_blue_line//state_signal_003_' + '0803_50.xlsx',
        encoding='gbk')


