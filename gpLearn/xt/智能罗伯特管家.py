# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/2 15:07
@Author  : zhangfang
@File    : 智能罗伯特管家.py
'''

import math
import pandas as pd
import numpy as np
# 本代码由可视化策略环境自动生成 2020年11月2日 15:01
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


# 回测引擎：初始化函数，只执行一次
def m3_initialize_bigquant_run(context):
    context.ranker_prediction = context.options['data'].read_df()
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    stock_count = 1

    context.stock_weights = T.norm([1 / math.log(i + 2) for i in range(0, stock_count)])

    context.max_cash_per_instrument = 1 / stock_count
    context.options['hold_days'] = 1

    context.sell_inst = []


# 回测引擎：每日数据处理函数，每天执行一次
def m3_handle_data_bigquant_run(context, data):
    import talib
    from datetime import timedelta
    # 按日期过滤得到今日的预测数据
    cur_date = data.current_dt
    d1 = timedelta(days=1)
    d100 = timedelta(days=80)
    start_date = (cur_date - d100).strftime('%Y-%m-%d')
    end_date = (cur_date - d1).strftime('%Y-%m-%d')
    ranker_prediction = context.ranker_prediction[context.ranker_prediction.date == cur_date.strftime('%Y-%m-%d')]

    # 上证指数
    big_prices = DataSource('bar1d_index_CN_STOCK_A').read(['000001.HIX'], start_date=start_date, end_date=end_date,
                                                           fields=['close'])['close']
    big_values = big_prices.values
    dif_big = big_values[-1] / big_values[-5]
    macd, signal, hist = talib.MACD(np.array(big_prices).astype('float64'), 12, 26, 9)

    if datetime.datetime(*map(int, context.start_date.split('-'))) > datetime.datetime(2018, 8, 5):
        print(data.current_dt.strftime(
            '%Y-%m-%d') + '-------------------------------------------------------------------------')
        print(ranker_prediction[:10])

    cash_avg = context.portfolio.portfolio_value / 2
    cash_for_buy = cash_avg
    stocks = list(context.portfolio.positions.keys())

    if context.trading_day_index % context.options['hold_days'] == 0 or len(stocks) == 0:
        buy_cash_weights = context.stock_weights
        max_count = len(buy_cash_weights)
        buy_instruments = list(ranker_prediction.instrument[:max_count])
        for i in stocks:
            #             if i.symbol not in buy_instruments:
            context.order_target(i, 0)
        #             dif_big > 0.96
        if hist[-1] - hist[-2] > -14 and dif_big > 0.96:
            for i, instrument in enumerate(buy_instruments):
                inst = context.symbol(instrument)
                #                 inst not in stocks
                if True:
                    cash = cash_for_buy * buy_cash_weights[i]
                    context.order_value(inst, cash)

    else:
        context.sell_inst = stocks
        for i in stocks:
            context.order_target(i, 0)


# 回测引擎：准备数据，只执行一次
def m3_prepare_bigquant_run(context):
    pass


# 回测引擎：每个单位时间开始前调用一次，即每日开盘前调用一次。
def m3_before_trading_start_bigquant_run(context, data):
    pass


m1 = M.instruments.v2(
    start_date='2015-01-01',
    end_date='2018-01-01',
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

m2 = M.use_datasource.v1(
    datasource_id='bigquant-shtshao-zhinengluobote',
    start_date='',
    end_date=''
)

m3 = M.trade.v4(
    instruments=m1.data,
    options_data=m2.data,
    start_date='',
    end_date='',
    initialize=m3_initialize_bigquant_run,
    handle_data=m3_handle_data_bigquant_run,
    prepare=m3_prepare_bigquant_run,
    before_trading_start=m3_before_trading_start_bigquant_run,
    volume_limit=0,
    order_price_field_buy='open',
    order_price_field_sell='close',
    capital_base=1000000,
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='真实价格',
    product_type='股票',
    plot_charts=True,
    backtest_only=False,
    benchmark=''
)