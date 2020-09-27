# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:15
# @Author  : zhangfang
from CTP.FutureClient.jz_FutureApi_lib import JzStrategy
from trading_future.future_singleton import Future
from jqdatasdk import *
import datetime
from configDB import *
import pandas as pd
import matplotlib.pyplot as plt
import time
auth(JOINQUANT_USER, JOINQUANT_PW)


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour >= 18:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]


if __name__ == "__main__":
    path = 'G:/trading/trading_report/'
    account = '21900576'
    ip = '127.0.0.1:7277'
    bars = 5
    porfolio = Future()
    today = datetime.date.today()
    calen = get_trade_days(count=bars)
    calen = list(calen)
    calen, next_tradeday, EndDate, StartDate, hq_last_date = get_date(calen, today)
    print(calen)
    hq_last_date = hq_last_date[:4] + hq_last_date[5:7] + hq_last_date[8:]
    today = datetime.date.strftime(today, '%Y%m%d')
    api = JzStrategy('ZhangFang', ip)

    ### 查询
    fund_df = api.req_fund()
    print(fund_df)
    fund_df = fund_df.loc[:, ['交易日', '当前保证金总额', '期货结算准备金', '上次结算准备金', '平仓盈亏', '持仓盈亏', '投资者帐号']]
    fund_df['静态权益'] = fund_df['上次结算准备金']
    fund_df['保证金'] = fund_df['当前保证金总额']
    fund_df['动态权益'] = fund_df['期货结算准备金']
    fund_df['保证金使用占比'] = fund_df['保证金'] / fund_df['动态权益']
    fund_pos = fund_df.loc[:, ['交易日', '动态权益', '保证金', '保证金使用占比', '平仓盈亏', '持仓盈亏', '静态权益', '投资者帐号']]
    fund_net = fund_pos['动态权益'].tolist()[0]
    fund_lastday = pd.read_excel(path + 'fund_' + account + '_' + hq_last_date + '.xlsx', index_col=0)
    fund_pos = pd.concat([fund_pos, fund_lastday])
    fund_pos.to_excel(path + 'fund_' + account + '_' + today + '.xlsx')
    net = fund_pos.loc[:, ['交易日', '静态权益']].rename(columns={'交易日': 'date'})
    net['date'] = net['date'].apply(lambda x: str(x))
    net = net.sort_values(['date'])
    net['权益'] = net['静态权益'].shift(-1).fillna(value=fund_net)
    net['净值曲线'] = net['权益'] / net['权益'].tolist()[0]
    net.to_excel(path + 'net_' + account + '_' + today + '.xlsx')
    net['date'] = pd.to_datetime(net['date'])
    title_str = '账户%s 净值曲线' % (account)
    net.set_index(['date']).ix[:, ['净值曲线']].plot()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(title_str)
    plt.savefig(path + 'fig/' + account + '_net_' + today + '.png')
    # plt.show()
    time.sleep(1)
    hold_df = api.req_hold()
    hold_pos = hold_df.loc[:, ['交易所', '交易日', '合约', '方向', '总仓', '上次结算价', '本次结算价', '保证金', '平仓盈亏', '持仓盈亏', '开仓成本', '持仓成本', '帐号']]
    hold_pos['symbol'] = hold_df['合约'].apply(lambda x: ''.join(filter(str.isalpha, x)))
    hold_pos['symbol'] = hold_pos['symbol'].apply(lambda x: x.upper())
    print(hold_pos)
    hold_pos = hold_pos.tail(6)
    contract_lst = hold_pos.symbol.tolist()
    print(contract_lst)
    VolumeMultiple_dict = porfolio.get_VolumeMultiple(contract_lst)
    VolumeMultiple_lst = []
    for symbol in contract_lst:
        VolumeMultiple_lst.append(VolumeMultiple_dict[symbol]['VolumeMultiple'])
    hold_pos['VolumeMultiple'] = VolumeMultiple_lst
    hold_pos['开仓均价'] = hold_pos['开仓成本'] / hold_pos['总仓'] / hold_pos['VolumeMultiple']
    hold_pos['持仓均价'] = hold_pos['持仓成本'] / hold_pos['总仓'] / hold_pos['VolumeMultiple']
    hold_pos = hold_pos.loc[:, ['交易日', '合约', '方向', '总仓', '上次结算价', '本次结算价', '保证金', '平仓盈亏', '持仓盈亏', '开仓均价', '持仓均价', '帐号', '开仓成本', '持仓成本', 'VolumeMultiple']]

    hold_pos['资金占比'] = hold_pos['保证金'] / fund_net
    hold_pos.to_excel(path + 'hold_' + account + '_' + today + '.xlsx')
    time.sleep(1)
    trades_df = api.req_trades()
    trades_df = trades_df.sort_values(['成交时间'])
    print(trades_df)
    trades_df.to_excel(path + 'trade_' + account + '_' + today + '.xlsx')

    account = '85030120'
    ip = '127.0.0.1:7278'
    bars = 5
    porfolio = Future()
    today = datetime.date.today()
    calen = get_trade_days(count=bars)
    calen = list(calen)
    calen, next_tradeday, EndDate, StartDate, hq_last_date = get_date(calen, today)
    hq_last_date = hq_last_date[:4] + hq_last_date[5:7] + hq_last_date[8:]
    today = datetime.date.strftime(today, '%Y%m%d')
    api = JzStrategy('ZhangFang', ip)

    ### 查询
    fund_df = api.req_fund()
    print(fund_df)
    fund_df = fund_df.loc[:, ['交易日', '当前保证金总额', '期货结算准备金', '上次结算准备金', '平仓盈亏', '持仓盈亏', '投资者帐号']]
    fund_df['静态权益'] = fund_df['上次结算准备金']
    fund_df['保证金'] = fund_df['当前保证金总额']
    fund_df['动态权益'] = fund_df['期货结算准备金']
    fund_df['保证金使用占比'] = fund_df['保证金'] / fund_df['动态权益']
    fund_pos = fund_df.loc[:, ['交易日', '动态权益', '保证金', '保证金使用占比', '平仓盈亏', '持仓盈亏', '静态权益', '投资者帐号']]
    fund_net = fund_pos['动态权益'].tolist()[0]
    fund_lastday = pd.read_excel(path + 'fund_' + account + '_' + hq_last_date + '.xlsx', index_col=0)
    fund_pos = pd.concat([fund_pos, fund_lastday])
    fund_pos.to_excel(path + 'fund_' + account + '_' + today + '.xlsx')
    net = fund_pos.loc[:, ['交易日', '静态权益']].rename(columns={'交易日': 'date'})
    net['date'] = net['date'].apply(lambda x: str(x))
    net = net.sort_values(['date'])
    net['静态权益'] = net['静态权益'].shift(-1).fillna(value=fund_net)
    net['净值曲线'] = net['静态权益'] / net['静态权益'].tolist()[0]
    net.to_excel(path + 'net_' + account + '_' + today + '.xlsx')
    net['date'] = pd.to_datetime(net['date'])
    title_str = '账户%s 净值曲线' % (account)
    net.set_index(['date']).ix[:, ['净值曲线']].plot()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(title_str)
    plt.savefig(path + 'fig/' + account + '_net_' + today + '.png')
    # plt.show()

    hold_df = api.req_hold()
    hold_pos = hold_df.loc[:,
               ['交易所', '交易日', '合约', '方向', '总仓', '上次结算价', '本次结算价', '保证金', '平仓盈亏', '持仓盈亏', '开仓成本', '持仓成本', '帐号']]
    hold_pos['symbol'] = hold_df['合约'].apply(lambda x: ''.join(filter(str.isalpha, x)))
    hold_pos['symbol'] = hold_pos['symbol'].apply(lambda x: x.upper())
    contract_lst = hold_pos.symbol.tolist()
    VolumeMultiple_dict = porfolio.get_VolumeMultiple(contract_lst)
    VolumeMultiple_lst = []
    for symbol in contract_lst:
        VolumeMultiple_lst.append(VolumeMultiple_dict[symbol]['VolumeMultiple'])
    hold_pos['VolumeMultiple'] = VolumeMultiple_lst
    hold_pos['开仓均价'] = hold_pos['开仓成本'] / hold_pos['总仓'] / hold_pos['VolumeMultiple']
    hold_pos['持仓均价'] = hold_pos['持仓成本'] / hold_pos['总仓'] / hold_pos['VolumeMultiple']
    hold_pos = hold_pos.loc[:,
               ['交易日', '合约', '方向', '总仓', '上次结算价', '本次结算价', '保证金', '平仓盈亏', '持仓盈亏', '开仓均价', '持仓均价', '帐号', '开仓成本', '持仓成本',
                'VolumeMultiple']]

    hold_pos['资金占比'] = hold_pos['保证金'] / fund_net
    hold_pos.to_excel(path + 'hold_' + account + '_' + today + '.xlsx')

    trades_df = api.req_trades()
    trades_df = trades_df.sort_values(['成交时间'])
    print(trades_df)
    trades_df.to_excel(path + 'trade_' + account + '_' + today + '.xlsx')
