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
    level = 5
    s_date = '2015-01-01'
    e_date = '2020-07-01'
    trd_num = 24
    fee = np.float(0.00015)
    # fold_ini_path = 'e://Strategy//MT4//'
    fold_ini_path = 'G://缠论//回测报告//'
    porfolio = Future()
    period_ini_lst = [5, 15, 30, 60, 240, 1440]
    period_lst_all = PowerSetsRecursive(period_ini_lst)
    period_lst_all = [i for i in period_lst_all if len(i) == 5 and 5 not in i]
    period_lst_all = [[15, 30, 60, 240]]
    code_trade_all = ['v', 'ap', 'sm', 'fu', 'zc', 'ma', 'rb', 'ta', 'hc', 'i', 'j', 'al', 'cu', 'ni', 'zn', 'jm','bu', 'p',
                      'pp', 'c', 'ag', 'pb', 'sf']

    code_lst = ['ap', 'ag', 'al', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
                'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta', 'm', 'ma']  # 所有品种32个
    # code_lst_5 = ['ap', 'j', 'rb', 'i', 'fu', 'sm', 'if', 'v', 'zn', 'pp', 'ni', 'pb']  # 所有5分钟夏普>0
    # code_lst_15 = ['v', 'sm', 'sf', 'ap', 'ni', 'j', 'i', 'if', 'hc', 'cu', 'al', 'pp', 'zc', 'rb', 'c', 'zn',
    #                'ag', 'pb', 'sc', 'sr', 'fu']  # 所有15分钟夏普>0
    # code_lst_30 = ['zc', 'v', 'ap', 'sm', 'if', 'al', 'rb', 'j', 'sc', 'fu', 'i', 'ta', 'sf', 'hc', 'pp']  # 所有30分钟夏普>0
    # code_lst_60 = ['ap', 'hc', 'j', 'rb', 'sc', 'al', 'ni', 'sf', 'fu', 'ta', 'zc', 'v',
    #                'bu', 'i', 'sm', 'ma', 'tf', 'zn']  # 所有60分钟夏普>0
    # code_lst_240 = ['al', 'cu', 'v', 'i', 'ma', 'j', 'zn', 'jm', 'fu', 'bu', 'rb',
    #                 'sm', 'ta', 'p', 'zc', 'hc', 'c', 'pp', 'if', 'ru', 'pb']  # 所有4小时夏普>0
    # code_lst_1440 = ['v', 'ma', 'fu', 'cu', 'j', 'au', 'cf', 'c', 'ta', 'pp', 'sf', 'ag', 'jm', 'sr', 'tf', 'if',
    #                  'hc', 'bu', 'zn', 'sm']  # 所有日级别夏普>0
    code_lst_5 = []  # 所有5分钟夏普>0
    # code_lst_15 = ['v', 'sm', 'sf', 'ni', 'j', 'i', 'if', 'hc', 'cu', 'al', 'pp', 'c', 'zn',
    #                'ag', 'pb', 'sc', 'sr']  # 所有15分钟夏普>0
    # code_lst_30 = ['zc', 'v', 'ap', 'sm', 'if', 'al', 'j', 'sc', 'i', 'pp']  # 所有30分钟夏普>0
    # code_lst_60 = ['ap', 'hc', 'j', 'rb', 'sc', 'al', 'ni', 'sf', 'fu', 'ta', 'zc',
    #                'i', 'ma', 'tf', 'zn']  # 所有60分钟夏普>0
    # code_lst_240 = ['al', 'cu', 'i', 'ma', 'j', 'zn', 'jm', 'fu', 'bu', 'rb',
    #                 'ta', 'p', 'if', 'ru', 'pb']  # 所有4小时夏普>0
    code_lst_1440 = []  # 所有日级别夏普>0

    code_lst_15 = ['ag', 'j', 'fu', 'zc', 'c', 'pb', 'ta']  # 所有15分钟夏普>0
    code_lst_30 = ['v', 'sm', 'rb']  # 所有30分钟夏普>0
    code_lst_60 = ['al', 'ap', 'j', 'hc', 'ni', 'fu', 'bu', 'zc']  # 所有60分钟夏普>0
    code_lst_240 = ['v', 'sm', 'cu', 'i', 'ma', 'zn', 'jm', 'bu', 'p', 'pp', 'c', 'sf', 'ta']  # 所有4小时夏普>0
    porfolio_lst = []
    signal_lst = []
    for trd_num in range(1, len(code_trade_all) + 1):
        code_trade_lst = code_trade_all[:trd_num]

        code_dict = {}
        code_dict['5'] = [i for i in code_lst_5 if i in code_trade_lst]
        code_dict['15'] = [i for i in code_lst_15 if i in code_trade_lst]
        code_dict['30'] = [i for i in code_lst_30 if i in code_trade_lst]
        code_dict['60'] = [i for i in code_lst_60 if i in code_trade_lst]
        code_dict['240'] = [i for i in code_lst_240 if i in code_trade_lst]
        code_dict['1440'] = [i for i in code_lst_1440 if i in code_trade_lst]

        for period_lst in period_lst_all:
            print(period_lst)
            chg_df_all = pd.DataFrame(columns=['date_time'])
            chg_name = []
            period_num_name_dict = {}
            for code in code_trade_lst:
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
                # print(profit_df)

                net_lst = profit_df.net.tolist()
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
                signal_row.append(len(period_name))
                signal_row.append(trd_num)
                signal_row.append('_'.join([str(i) for i in period_name]))
                signal_row.append(fee)
                signal_row.append(sharpe_ratio)
                signal_row.append(ann_return)
                signal_row.append(max_drawdown)
                signal_row.append(s_date)
                signal_row.append(e_date)
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
                # chg_df_ = chg_df_.rename(columns={'chg': 'chg_' + code})
                chg_df_all = chg_df_all.merge(chg_df_, on=['date_time'], how='outer')
            chg_df_all = chg_df_all.fillna(value=0)
            chg_df = chg_df_all.sort_values(['date_time']).set_index(['date_time'])
            # period_num_name = ['period_num_' + code for code in code_trade_lst]
            period_sum = sum([period_num_name_dict[code[4:]] for code in chg_name])
            chg_df['chg'] = chg_df[chg_name].sum(axis=1) / period_sum
            chg_df = chg_df.fillna(value=0)
            chg_df['net'] = 1 + chg_df['chg'].cumsum()
            chg_df = chg_df.reset_index(drop=False)
            chg_df['date_time'] = pd.to_datetime(chg_df['date_time'])
            chg_df = chg_df.set_index(['date_time'])
            sharpe_ratio = yearsharpRatio(chg_df['net'].tolist(), 1)
            ann_return = annROR(chg_df['net'].tolist(), 1)
            max_drawdown = maxRetrace(chg_df['net'].tolist(), 1)
            porfolio_row = []
            porfolio_row.append(len(code_trade_lst))
            porfolio_row.append('_'.join([str(i) for i in period_lst]))
            porfolio_row.append(fee)
            porfolio_row.append(sharpe_ratio)
            porfolio_row.append(ann_return)
            porfolio_row.append(max_drawdown)
            porfolio_row.append(s_date)
            porfolio_row.append(e_date)
            porfolio_lst.append(porfolio_row)

            chg_df.ix[:, ['net']].plot()
            title_str = '品种%s个 周期%sm sharp %.2f annRet %.2f maxRetrace %.2f' % (
                str(len(code_trade_lst)),  '_'.join([str(i) for i in period_lst]), sharpe_ratio, 100 * ann_return,
                100 * max_drawdown)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.title(title_str)
            plt.savefig(fold_ini_path + 'fig/' + str(len(code_trade_lst)) + '_' + str(period) + 'm' + '_fee_' +
                        str(int(np.around(fee*100000, 0))) + '.png')
            plt.show()

    porfolio_state = pd.DataFrame(porfolio_lst, columns=['品种数', 'period', 'fee', 'sharpe_ratio', 'ann_return',
                                                         'max_drawdown', 's_date', 'e_date'])
    porfolio_state.to_excel(fold_ini_path + 'state_blue_line//state_porfolio_all_period_' + e_date + '实盘1.xlsx', encoding='gbk')

    signal_state = pd.DataFrame(signal_lst, columns=['品种', 'period_num', 'trd_num', 'period', 'fee', 'sharpe_ratio', 'ann_return',
                                                         'max_drawdown', 's_date', 'e_date'])
    signal_state.to_excel(
        fold_ini_path + 'state_blue_line//state_signal_all_period实盘1' + '.xlsx',
        encoding='gbk')


