'''
交易模块，编写交易代码。
用来交易的数据。
'''
import math
import time
import datetime
import pandas as pd
import numpy as np
from logger import logger
from get_data import read_removed_st_mkt, get_last_trade_date, read_benchmark
import param
from copy import deepcopy
from get_data import planedorders, read_y_values

'''
第一部分：信号生成部分的代码写法
initialize: 为初始化方法，准备所有的数据，参数。只初始一次。
before_trading_start：盘前函数处理，每日8.45掉起一次。准备交易信息号。
其他方法都为准备方法。
'''


def initialize_wf(context):
    '''
    初始化数据，设置最初交易信息
    '''
    # 初始化各种参数信息，
    init_args(context=context)


def init_args(context):
    '''
    初始化各种参数信息
    1.初始化，手续费，日志，买入股票列表，卖出股票列表，权重值
    :param context:
    :return:
    '''

    arg_option = param.backtest_args_option

    Log = logger(set_level=arg_option['log_set_level'], file_path=arg_option['log_file_path'],
                 use_console=True).InItlogger()
    context.Log = Log  # 将日志对象放入全局变量中

    for key, value in arg_option.items():
        setattr(context.__class__, key, value)

    Log.info('初始化init函数，初始化所需参数：' + str(arg_option))

    # 设置买入的股票数量，这里买入预测股票列表排名靠前的5只
    stock_count = context.buy_size
    context.stock_count = stock_count
    # 每只的股票的权重，如下的权重分配会使得靠前的股票分配多一点的资金，[0.339160, 0.213986, 0.169580, ..]
    high_list = [1 / math.log(i + 2) for i in range(0, stock_count)]
    data = sum(high_list)
    stock_weights = [t / data for t in high_list]
    context.stock_weights = stock_weights


def get_all_portfolio(context):
    '''获取总账户持仓信息'''
    Log = context.Log
    hold_portfolio = context.get_portfolio()
    Log.debug('xuntou:读取总账户持仓信息:' + str(hold_portfolio))
    return hold_portfolio


def get_all_positions(context):
    '''获取股票持仓信息'''

    Log = context.Log
    all_positions = context.get_positions()
    Log.debug('读取股票仓信息:' + str(all_positions))
    return all_positions


# ---------下面是盘前函数相关方法--------------
def the_day_buy(context):
    """
    计算当日要买的股票，初选，只是确定最初的交易信号
        1.从全局变量中获取已传入的y值信息
        2.获取上一个交易日
        3.获取上一个交易日的y值前buy_size个股票
        4.添加到股买入股票列表中
        5.添加到全局变量中
    """
    Log = context.Log
    # today 是今天的上一个交易日
    today = context.get_datetime().strftime('%Y-%m-%d')
    if not param.is_backtest:
        buy_markets = planedorders(today, today, param.strategy_map[context.strategy_name])
    else:
        buy_markets = planedorders(today, today, param.strategy_map[context.strategy_name])
    Log.info('获取当前时间：' + str(today))
    Log.info('查找当日要买的股票：' + str(buy_markets))
    Log.info('--开始计算当日要买的股票--')
    # 计算买入的金额
    cash_avg = context.get_portfolio()['portfolio_value'] / context.hold_days

    if param.is_backtest:
        cash_for_buy = min(context.get_portfolio()['cash'], 1 * cash_avg)
    else:
        cash_for_buy = min(context.get_portfolio()['cash'], 1 * cash_avg) / param.strategy_num / 5
    # cash_for_buy = min(context.get_portfolio()['cash'], 1 * cash_avg)
    buy_cash_weights = context.stock_weights
    # 买入时的各种信息
    buy_info = {}
    # 初始股票仓位是0,进行初选，买入；下面就是循环进行初始化
    if buy_markets:
        for i, buy_market in enumerate(buy_markets):
            if buy_market in ['300090.SZ', '600240.SH', '300362.SZ', '600069.SH', '300028.SZ', '300156.SZ']:
                continue
            close_cul = float(context.current([buy_market], ['close']))
            print(close_cul, cash_for_buy, buy_cash_weights[i])
            buy_num = int((cash_for_buy * buy_cash_weights[i] // (100 * close_cul)) * 100)
            buy_info[buy_market] = {'generate_time': today, 'buy_ratio': 0, 'buy_time': None,
                                    'buy_cash': cash_for_buy * buy_cash_weights[i], 'buy_num': buy_num}
    Log.info('当日要买的股票信息为:' + str(buy_info))
    # 计算要买入的数量（整100的倍数）
    context.buy_info = buy_info


def the_day_sell(context):
    """
    计算好昨日要卖哪些
        1.获取现有的持仓信息
        2.添加到字典中
        3.将需要卖出的股票信息添加全局变量中
    """
    Log = context.Log
    now = context.get_datetime().strftime('%Y-%m-%d')
    # 产生要卖出的股票列表
    sell_markets = {}
    # 获取持仓的股票数据 dict
    hold_markets = get_all_positions(context=context)
    # 计算今天和前一个交易日
    today = context.get_datetime().strftime('%Y-%m-%d')
    last_trade_date = get_last_trade_date(today=today, context=context)
    if hold_markets:
        # 将持仓都放入卖出集合中
        for hold_market, hold in hold_markets.items():
            if hold_market[0] == '6':
                hold_market = hold_market + '.SH'
            else:
                hold_market = hold_market + '.SZ'
            sell_markets[hold_market] = {'time': now, 'amount': hold['amount'], 'reason': '第二日卖出', 'selled_sign': 0,
                                         'mean_lines': [], 'close_info': []}  # 0代表没卖，1代表卖出已经完成

    Log.info('当日要卖的股票信息为:' + str(sell_markets))
    # 设定卖出时间
    context.sell_markets = sell_markets


def risk_manage(context):
    """
    交易之前，盘前处理，风险管理
        1.读取风控文件
        if 清仓：
            要买的股票置空

        else：
            深度复制 要买入的股票列表【】
            for  股票  in 股票列表：
                if  （如果 股票 在风控list中） or  不在股票池中   or 股票代码开头为688：
                    从要买的股票列表中剔除掉此股

            将剔除后的股票列表添加到全局变量中【】

    """
    # 不买st
    Log = context.Log
    mkt_sts, clean_all = read_removed_st_mkt(positon='file')
    # 如果clean_all标志是1，则全部卖出；今日要买入的股票也清零
    if clean_all == 1:
        context.buy_info = []
        return
    else:
        no_deal = []
        copy_info = deepcopy(context.buy_info)
        # 避免计算的出错，复制一份出来
        for buy_market in context.buy_info:
            # 不买风控的，不买没订阅的，不买科创板。
            if (buy_market in mkt_sts) or (str(buy_market).startswith("688")):
                Log.debug('执行风险管理，剔除的股票为：' + str(buy_market))
                no_deal.append(buy_market)
                del copy_info[buy_market]

        context.buy_info = copy_info
        Log.info('风控不交易的股票有：' + str(no_deal))


def before_trading_start_wf(context):
    '''
    盘前处理函数
    '''
    Log = context.Log
    Log.info('-----------------盘前函数处理开始----------------------')
    # 获取当日要买的股票
    the_day_buy(context=context)
    ##获取当日要卖出的股票
    the_day_sell(context=context)
    ##经过风险管理
    risk_manage(context=context)
    Log.info('-----------------盘前函数处理结束----------------------')


def cancel_yestoday_order(context):
    '''如果昨天的下单没有成交  ，取消所有的下单'''
    for orders in context.get_open_orders().values():
        # 循环，撤销订单
        for _order in orders:
            context.cancel_order(_order)
