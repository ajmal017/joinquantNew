# -*- coding: utf-8 -*-
'''
@Time    : 2020/10/30 16:36
@Author  : zhangfang
@File    : ST_warning.py
'''
from gpLearn.xt.strategy_state import planedorders, get_strategy_signal_history
from configDB import *
from jqdatasdk import *
import pandas as pd
from gpLearn.xt.word_pdf import FormatConvert, SendMessage
import datetime
auth(JOINQUANT_USER, JOINQUANT_PW)


if __name__ == '__main__':
    EndDate = datetime.datetime.today().strftime('%Y-%m-%d')
    strategy_name_lst = ['AI智能驱动', '智能罗伯特管家', '时代先锋', '淘利阿尔法1号']
    signal = get_strategy_signal_history(strategy_name_lst, '2020-10-30', '2020-10-30')
    print(signal)
    buy_signal = signal[signal['trade_side'] == 1]
    ret = []
    for code in buy_signal.symbol.tolist():
        a = finance.run_query(query(finance.STK_LIST).filter(finance.STK_LIST.code == normalize_code(code))\
                              .limit(100))[['code', 'name', 'state']]
        print(a)
        ret.append(a)
    ret = pd.concat(ret).rename(columns={'state': '上市状态', 'code': '股票代码', 'name': '证券名称'})
    print(ret)
    subject = f'信号上市状态{EndDate}'

    password = '9eFzgacCkDMUpPP6'
    sender = 'aiquant@ai-quants.com'
    # 收件人为多个收件人
    receiver = ["zhangfang@ai-quants.com"]
    # receiver = ['aiquant@ai-quants.com']
    send = SendMessage(sender, password)
    send.send_email1(ret, subject, sender, receiver)