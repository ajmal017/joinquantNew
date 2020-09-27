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
    ip = '127.0.0.1:1114'
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
    fund_df['动态权益'] = fund_df['上次结算准备金'] + fund_df['平仓盈亏'] + fund_df['持仓盈亏']
    fund_df['保证金使用占比'] = fund_df['保证金'] / fund_df['动态权益']
    fund_pos = fund_df.loc[:, ['交易日', '动态权益', '保证金', '保证金使用占比', '平仓盈亏', '持仓盈亏', '静态权益', '投资者帐号']]
    fund_net = fund_pos['动态权益'].tolist()[0]
    fund_lastday = pd.read_excel(path + 'fund_' + account + '_' + hq_last_date + '.xlsx', index_col=0)
    fund_pos = pd.concat([fund_pos, fund_lastday])
    fund_pos.to_excel(path + 'fund_' + account + '_' + today + '.xlsx')
    net = fund_pos.loc[:, ['交易日', '静态权益']].rename(columns={'交易日': 'date'})
    net['date'] = net['date'].apply(lambda x: str(x))
    net = net.sort_values(['date'])
    net['净值曲线'] = net['静态权益'].shift(-1).fillna(value=fund_net)
    net['净值曲线'] = net['净值曲线'] / net['净值曲线'].tolist()[0]
    net.to_excel(path + 'net_' + account + '_' + today + '.xlsx')
    net['date'] = pd.to_datetime(net['date'])
    title_str = '账户%s 净值曲线' % (account)
    net.set_index(['date']).ix[:, ['净值曲线']].plot()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(title_str)
    plt.savefig(path + 'fig/' + account + '_net_' + today + '.png')
    # plt.show()

    hold_df = api.req_hold()
    hold_pos = hold_df.loc[:, ['交易所', '交易日', '合约', '方向', '总仓', '上次结算价', '本次结算价', '保证金', '平仓盈亏', '持仓盈亏', '开仓成本', '持仓成本', '帐号']]
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
    hold_pos = hold_pos.loc[:, ['交易日', '合约', '方向', '总仓', '上次结算价', '本次结算价', '保证金', '平仓盈亏', '持仓盈亏', '开仓均价', '持仓均价', '帐号', '开仓成本', '持仓成本', 'VolumeMultiple']]

    hold_pos['资金占比'] = hold_pos['保证金'] / fund_net
    hold_pos.to_excel(path + 'hold_' + account + '_' + today + '.xlsx')

    trades_df = api.req_trades()
    trades_df = trades_df.sort_values(['成交时间'])
    trades_df.to_excel(path + 'trade_' + account + '_' + today + '.xlsx')

    # orders_df = api.req_orders()
    #
    # cancelable_df = api.req_cancelable()
    #
    # ### 交易部分：
    # #### 1、开启enable_trade后才会交易（api.enable_trade=True）
    # #### 2、交易：配置期货代码ocde、价格price、成交量vol、多空long/short、开平open/close/closetoday、策略id(如'test_001','test002')
    # #### 3、撤单：需要从req_orders()中获取FrontID、SessionID、orderRef加上期货代码才可以撤单
    #
    # # api.enable_trade = True
    # ### 下单代码
    # code = 'rb2101'
    # # 价格
    # price = 3325
    # # 成交量
    # vol = 1
    # # 多空 long short
    # direction = 'long'
    # # 开平 open close
    # open_close = 'open'
    # strategy_name = ''
    # # return_order = api.send_order(code,price,vol,direction,open_close)
    # return_order = api.send_order(code, price, vol, direction, open_close, strategy_name=strategy_name)
    # ### 下单
    # # 全撤
    # api.enable_trade = True
    # cancelable_df = api.req_cancelable()
    # for i, cancel_orders in cancelable_df.iterrows():
    #     FrontID = cancel_orders['FrontID']
    #     SessionID = cancel_orders['SessionID']
    #     orderRef = cancel_orders['报单引用']
    #     code = cancel_orders['合约']
    #     api.order_cancel(FrontID, SessionID, orderRef, code)
