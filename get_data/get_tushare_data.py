# -*- coding: utf-8 -*-
'''
@Time    : 2020/9/30 15:21
@Author  : zhangfang
@File    : get_tushare_data.py
'''
import tushare as ts
pro = ts.pro_api('9010f5cb499913c26a48c5fd3ac56dc41018de47d40daf82960f4c9b')
df = pro.fut_basic(exchange='CZCE', fut_tuye=2)
# df = pro.daily(ts_code='000001.SZ', start_date='20200921', end_date='20201114')
print(df)
