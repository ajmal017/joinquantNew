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
    fold_path = 'e://Strategy//MT4//'
    df = pd.read_html(fold_path + '蓝线笔_红线反转确认_蓝线反转平仓_200627//sm.htm', encoding='gbk')
    print(df)
    print(len(df))
    state = df[0]
    trade = df[1]
    print(df[1])
    print(state.iloc[14, 3])
    s = state.iloc[14, 3]
    print(s)
    print(s.split())
    print(s.split('('))
    print(s.split(')'))
    # state.to_csv(fold_path + 'sm.csv', encoding='gbk')
    ret = {}
    ret['交易品种'] = [state.iloc[0, 2].split('.')[0]]
    ret['周期'] = [state.iloc[1, 2].split()[0] + state.iloc[1, 2].split()[1]]
    ret['start_time'] = [state.iloc[1, 2].split()[2]]
    ret['end_time'] = [state.iloc[1, 2].split()[5]]
    ret['复盘模型'] = [state.iloc[2, 2]]
    ret['K线数量'] = [state.iloc[5, 1]]
    ret['盈利比'] = [state.iloc[10, 1]]
    ret['交易次数'] = [state.iloc[13, 1]]
    ret['盈利次数'] = [int(state.iloc[14, 3].split()[0])]
    ret['平均盈利'] = [state.iloc[16, 3]]
    ret['平均亏损'] = [state.iloc[16, 5]]
    ret['点差'] = [state.iloc[8, 5]]
    ret = pd.DataFrame(ret).T

    print(ret)
