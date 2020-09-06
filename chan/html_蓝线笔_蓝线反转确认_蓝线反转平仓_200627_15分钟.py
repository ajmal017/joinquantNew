# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:15
# @Author  : zhangfang
import pandas as pd
import html5lib


def PowerSetsRecursive(items):
    # 求集合的所有子集
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


if __name__ == "__main__":
    mode = '蓝线笔_蓝线反转确认_蓝线反转平仓_200627_15分钟'
    fold_path = 'e://Strategy//MT4//' + mode + '//'
    code_lst = ['ag', 'al', 'ap', 'cf', 'cu', 'fu', 'i', 'j', 'ni', 'pb', 'pp', 'rb', 'sc', 'tf', 'v', 'zc', 'zn', 'c',
                'if', 'sf', 'p', 'hc', 'au', 'jm', 'sm', 'ru', 'bu', 'oi', 'sr', 'ta']
    state_df = []
    for code in code_lst:
        html = pd.read_html(fold_path + code + '.htm', encoding='gbk')
        state = html[0]
        trade = html[1]
        ret = {}
        ret['symbol'] = [state.iloc[0, 2].split('.')[0]]
        ret['tm'] = [state.iloc[1, 2].split()[0] + state.iloc[1, 2].split()[1]]
        ret['start_time'] = [state.iloc[1, 2].split()[2]]
        ret['end_time'] = [state.iloc[1, 2].split()[5]]
        ret['复盘模型'] = [state.iloc[2, 2]]
        ret['K线数量'] = [state.iloc[5, 1]]
        ret['盈利比'] = [state.iloc[10, 1]]
        ret['trading_times'] = [int(state.iloc[13, 1])]
        ret['盈利次数'] = [int(state.iloc[14, 3].split()[0])]
        ret['平均盈利'] = [float(state.iloc[16, 3])]
        ret['平均亏损'] = [float(state.iloc[16, 5])]
        ret['点差'] = [int(state.iloc[8, 5])]
        ret = pd.DataFrame(ret)
        state_df.append(ret)
    state_df = pd.concat(state_df)
    state_df['胜率'] = state_df['盈利次数'] / state_df['trading_times']
    state_df['盈亏比'] = -state_df['平均盈利'] / state_df['平均亏损']
    state_df['开仓中枢过滤'] = '无中枢过滤'
    state_df['开仓确认'] = '蓝线反转'
    state_df['平仓有无背离'] = '无背离'
    state_df['模型'] = mode
    print(state_df)

    state_df.to_excel(fold_path + 'state_winR_odd.xlsx', encoding='gbk')

