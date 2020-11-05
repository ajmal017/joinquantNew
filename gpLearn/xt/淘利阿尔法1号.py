# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/2 15:05
@Author  : zhangfang
@File    : 淘利阿尔法1号.py
'''

import math
import pandas as pd
# 本代码由可视化策略环境自动生成 2020年11月2日 14:58
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


# 回测引擎：初始化函数，只执行一次
def m3_initialize_bigquant_run(context):
    # 加载预测数据
    context.ranker_prediction = context.options['data'].read_df()

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 预测数据，通过options传入进来，使用 read_df 函数，加载到内存 (DataFrame)
    # 设置买入的股票数量，这里买入预测股票列表排名靠前的5只
    stock_count = 1
    # 每只的股票的权重，如下的权重分配会使得靠前的股票分配多一点的资金，[0.339160, 0.213986, 0.169580, ..]
    context.stock_weights = T.norm([1 / math.log(i + 2) for i in range(0, stock_count)])
    # 设置每只股票占用的最大资金比例
    context.max_cash_per_instrument = 0.99
    context.options['hold_days'] = 1


# 回测引擎：每日数据处理函数，每天执行一次
def m3_handle_data_bigquant_run(context, data):
    # 获取当日日期
    today = data.current_dt.strftime('%Y-%m-%d')
    stock_hold_now = [equity.symbol for equity in context.portfolio.positions]
    # 大盘风控模块，读取风控数据
    benckmark_risk = context.benckmark_risk[today]
    context.symbol
    # 当risk为1时，市场有风险，全部平仓，不再执行其它操作
    if benckmark_risk > 0:
        for instrument in stock_hold_now:
            context.order_target(symbol(instrument), 0)
        print(today, '大盘风控止损触发,全仓卖出')
        return

    # 按日期过滤得到今日的预测数据
    ranker_prediction = context.ranker_prediction[
        context.ranker_prediction.date == data.current_dt.strftime('%Y-%m-%d')]

    # 1. 资金分配
    # 平均持仓时间是hold_days，每日都将买入股票，每日预期使用 1/hold_days 的资金
    # 实际操作中，会存在一定的买入误差，所以在前hold_days天，等量使用资金；之后，尽量使用剩余资金（这里设置最多用等量的1.5倍）
    is_staging = context.trading_day_index < context.options['hold_days']  # 是否在建仓期间（前 hold_days 天）
    cash_avg = context.portfolio.portfolio_value / context.options['hold_days']
    cash_for_buy = min(context.portfolio.cash, (1 if is_staging else 1.5) * cash_avg)
    cash_for_sell = cash_avg - (context.portfolio.cash - cash_for_buy)
    positions = {e.symbol: p.amount * p.last_sale_price
                 for e, p in context.perf_tracker.position_tracker.positions.items()}

    # 2. 生成卖出订单：hold_days天之后才开始卖出；对持仓的股票，按机器学习算法预测的排序末位淘汰
    if not is_staging and cash_for_sell > 0:
        equities = {e.symbol: e for e, p in context.perf_tracker.position_tracker.positions.items()}
        instruments = list(reversed(list(ranker_prediction.instrument[ranker_prediction.instrument.apply(
            lambda x: x in equities and not context.has_unfinished_sell_order(equities[x]))])))
        # print('rank order for sell %s' % instruments)
        for instrument in instruments:
            context.order_target(context.symbol(instrument), 0)
            cash_for_sell -= positions[instrument]
            if cash_for_sell <= 0:
                break

    # 3. 生成买入订单：按机器学习算法预测的排序，买入前面的stock_count只股票
    buy_cash_weights = context.stock_weights
    buy_instruments = list(ranker_prediction.instrument[:len(buy_cash_weights)])
    max_cash_per_instrument = context.portfolio.portfolio_value * context.max_cash_per_instrument
    for i, instrument in enumerate(buy_instruments):
        cash = cash_for_buy * buy_cash_weights[i]
        if cash > max_cash_per_instrument - positions.get(instrument, 0):
            # 确保股票持仓量不会超过每次股票最大的占用资金量
            cash = max_cash_per_instrument - positions.get(instrument, 0)
        if cash > 0:
            context.order_value(context.symbol(instrument), cash)


# 回测引擎：准备数据，只执行一次
def m3_prepare_bigquant_run(context):
    pass


# 回测引擎：每个单位时间开始前调用一次，即每日开盘前调用一次。
def m3_before_trading_start_bigquant_run(context, data):
    # 在数据准备函数中一次性计算每日的大盘风控条件相比于在handle中每日计算风控条件可以提高回测速度
    # 多取50天的数据便于计算均值(保证回测的第一天均值不为Nan值)，其中context.start_date和context.end_date是回测指定的起始时间和终止时间
    start_date = (pd.to_datetime(context.start_date) - datetime.timedelta(days=50)).strftime('%Y-%m-%d')
    df = DataSource('bar1d_index_CN_STOCK_A').read(start_date=start_date, end_date=context.end_date, fields=['close'])
    benckmark_data = df[df.instrument == '399015.ZIX']
    # 计算上证指数5日涨幅
    benckmark_data['ret5'] = benckmark_data['close'] / benckmark_data['close'].shift(1) - 1
    # 计算大盘风控条件，如果5日涨幅小于-4%则设置风险状态risk为1，否则为0
    benckmark_data['risk'] = np.where(benckmark_data['ret5'] < -0.0075, 1, 0)
    # 修改日期格式为字符串(便于在handle中使用字符串日期索引来查看每日的风险状态)
    benckmark_data['date'] = benckmark_data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # 设置日期为索引
    benckmark_data.set_index('date', inplace=True)
    # 把风控序列输出给全局变量context.benckmark_risk
    context.benckmark_risk = benckmark_data['risk']


m1 = M.instruments.v2(
    start_date='2019-01-01',
    end_date='2019-11-01',
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

m4 = M.use_datasource.v1(
    datasource_id='bigquant-shtshao-taoli',
    start_date='',
    end_date=''
)

m3 = M.trade.v4(
    instruments=m1.data,
    options_data=m4.data,
    start_date='',
    end_date='',
    initialize=m3_initialize_bigquant_run,
    handle_data=m3_handle_data_bigquant_run,
    prepare=m3_prepare_bigquant_run,
    before_trading_start=m3_before_trading_start_bigquant_run,
    volume_limit=0.025,
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