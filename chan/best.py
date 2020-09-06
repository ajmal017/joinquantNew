# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 14:19
# @Author  : zhangfang
import pandas as pd


if __name__ == "__main__":
    fold_ini_path = 'e://Strategy//MT4//'
    # fold_ini_path = 'G://缠论//回测报告//'
    signal_state = pd.read_excel(fold_ini_path + 'state_blue_line//state_signal_003_0803' + '' + '.xlsx',
        encoding='gbk', index_col=0)
    print(signal_state)
    signal_state = signal_state.drop_duplicates(['品种', 'period', 's_date', 'e_date'], keep='first').dropna()
    print(signal_state)

    lst = []
    for method, group in signal_state.groupby(['品种', 'period']):
        row = []
        row.append(method[0])
        row.append(method[1])
        row.append(len(group[group['sharpe_ratio'] > 0]) / len(group))
        lst.append(row)
    df = pd.DataFrame(lst, columns=['品种', 'period', 'winR'])
    df.to_excel(fold_ini_path + 'state_blue_line//state_signal_003_0803' + '_select.xlsx',
        encoding='gbk')



    # signal_state.to_excel(
    #     fold_ini_path + 'state_blue_line//state_signal_缠论003' + 'dropduplicate0729.xlsx',
    #     encoding='gbk')