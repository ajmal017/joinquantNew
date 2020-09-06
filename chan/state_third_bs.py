# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:15
# @Author  : zhangfang
import pandas as pd
import numpy as np


def PowerSetsRecursive(items):
    # 求集合的所有子集
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


if __name__ == "__main__":
    level = 1
    s_date = '2015-01-01'
    e_date = '2020-01-01'
    fee = np.float(0.00015)
    fold_ini_path = 'e://Strategy//MT4//state_blue_line//'
    state_singal = pd.read_excel(
        fold_ini_path + 'state_singal_symbol_' + str(int(np.around(100000 * fee, 0))) + '.xlsx',
        encoding='gbk', index_col=0)
    print(state_singal)
    state_singal['总盈利'] = state_singal['平均盈利'] * state_singal['盈利次数']
    state_singal['总亏损'] = -state_singal['平均亏损'] * (state_singal['trading_times'] - state_singal['盈利次数'])
    porfolio_state = pd.read_excel(
        fold_ini_path + 'state_porfolio_' + str(int(np.around(100000 * fee, 0))) + '.xlsx',
        encoding='gbk', index_col=0)

    method_name = ['tm', 'symbol']  # ['tm', 'symbol']
    method_name_all = PowerSetsRecursive(method_name)
    method_name_all = [i for i in method_name_all if len(i) != 0]
    # print(method_name_all)

    lst = []
    for method_name1 in method_name_all:
        for method, group in state_singal.groupby(method_name1):

            rw = []
            rw.append(method)
            rw.append(group['盈利次数'].sum() / group['trading_times'].sum())
            rw.append(group['总盈利'].sum() / group['盈利次数'].sum() / (group['总亏损'].sum()/(group['trading_times'].sum()-group['盈利次数'].sum())))
            rw.append(group['trading_times'].mean())
            rw.append(group['sharp'].mean())
            rw.append(group['ann_return'].mean())
            rw.append(group['max_drawdown'].mean())
            rw.append(len(group[group['sharp'] > 0])/len(group))
            lst.append(rw)
    ret = pd.DataFrame(lst, columns=['模型', '平均胜率', '平均盈亏比', '平均交易次数', 'sharp', 'ann_return', 'max_drawdown', '盈利品种占比'])
    print(ret)
    ret.to_excel(fold_ini_path + 'state_' + '_'.join(method_name) + '.xlsx')

    symbol_max = state_singal.groupby(['symbol'])['sharp'].max()
    print(symbol_max)
    symbol_max = []
    for code, group in state_singal.groupby(['symbol']):
        symbol_max.append(group[group['sharp'] == group['sharp'].max()])
    symbol_max = pd.concat(symbol_max)
    print(symbol_max)
    symbol_max.to_excel(fold_ini_path + 'symbol_max_tm' + '.xlsx')

