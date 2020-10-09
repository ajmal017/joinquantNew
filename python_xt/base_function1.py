# encoding:gbk
import pandas as pd
import numpy as np
import param
from functools import partial
from logger import logger
import math
import time
import datetime
from copy import deepcopy
import os
import pickle as pk
import codecs
import requests
import json


def init(ContextInfo):
    # ContextInfo.get_trade_detail_data = get_trade_detail_data
    initialize(ContextInfo, 'AI智能驱动')
    ContextInfo.Log = []
    ContextInfo.get_trade_detail_data = []


def handlebar(ContextInfo):
    if ContextInfo.get_datetime().strftime('%Y-%m-%d %H:%M:%S') > time.strftime('%Y-%m-%d', time.localtime(
            time.time())) + ' 09:30:00':
        print(ContextInfo.get_datetime())
        # ContextInfo.get_trade_detail_data = get_trade_detail_data
        # ContextInfo.timetag_to_datetime = timetag_to_datetime
        # ContextInfo.passorder = passorder
        handlebar_xt(ContextInfo)
        ContextInfo.Log = []


# ContextInfo.get_trade_detail_data = []
# ContextInfo.timetag_to_datetime = []
# ContextInfo.passorder = []

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


def initialize(ContextInfo, strategy_name):
    '''
    初始化交易引擎
    :return:
    '''
    ContextInfo.strategy_name = strategy_name
    # ContextInfo.get_trade_detail_data = get_trade_detail_data
    if len(get_stocklist(ContextInfo)) > 0:
        ContextInfo.set_universe(get_stocklist(ContextInfo))
    ContextInfo.set_account(param.accountid)
    ContextInfo.get_datetime = partial(current_dt, ContextInfo)
    ContextInfo.order = partial(order, ContextInfo)
    ContextInfo.get_order = partial(get_order, ContextInfo)
    ContextInfo.current = partial(current, ContextInfo)
    ContextInfo.can_trade = partial(can_trade, ContextInfo)
    ContextInfo.get_positions = partial(get_positions, ContextInfo)
    ContextInfo.get_portfolio = partial(get_portfolio, ContextInfo)
    ContextInfo.symbol = symbol
    ContextInfo.start_buy_list = []
    initialize_wf(ContextInfo)
    ContextInfo.set_commission(0, param.commissionList)
    ContextInfo.start = param.start_date[0:10] + ' 09:00:00'
    ContextInfo.end = param.end_date[0:10] + ' 15:00:00'
    # 仅回测可用
    if param.is_backtest:
        ContextInfo.capital = param.capital_base
    else:
        # ContextInfo.capital = param.capital_base
        pass


def handlebar_xt(ContextInfo):
    '''
    事件驱动-主逻辑函数
    :return:
    '''
    ###模型入参数z
    arg_option = param.backtest_args_option
    for key, value in arg_option.items():
        setattr(ContextInfo.__class__, key, value)
    ContextInfo.Log = logger(set_level=ContextInfo.log_set_level, file_path=ContextInfo.log_file_path).InItlogger()

    print("--------------")
    print(current_dt(ContextInfo).strftime('%Y-%m-%d'))
    if param.is_backtest:
        if ContextInfo.get_datetime().strftime('%H:%M:%S') == '09:30:00':
            before_trading_start_wf(ContextInfo)
    else:
        if ContextInfo.get_datetime().strftime('%H:%M:%S') == '09:30:00':
            before_trading_start_wf(ContextInfo)

    handle_data(ContextInfo, ContextInfo)


def planedorders(s_date, e_date, unique_id):
    url = 'https://aicloud.citics.com/papertrading/api/obtain_orders'
    data = {
        # 可选，不填时读取所有策略
        'unique_id': unique_id,  # 'pt461a4de0cedf11e995950a580a81060a' ,
        # 必填，<API文档>中的API_KEY
        'api_key': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJleHAiOjcyNTgwODk2MDAsInVwZGF0ZV90aW1lIjoiMjAyMC0wOC0yOSAxNDowMDo1NCIsInVzZXJuYW1lIjoiYnFhZG0xIn0.mHwdmeHTW7vchJjKt1b_iiKwbJ8L-Soq6wy0g607ri4',
        'start_date': s_date[:10],
        'end_date': e_date[:10],
        'username': 'bqadm1',
    }
    r = requests.post(url=url, data=data)
    res_dict = json.loads(r.text)
    print(res_dict)
    if res_dict['code'] != 200:
        print('获取信号失败，code： %s' % res_dict['code'])
        return
    else:
        if len(res_dict['data']['order_list']) == 0:
            print('当日无信号')
            return
        else:
            df = pd.DataFrame.from_dict(res_dict['data']['order_list'])[['run_date', 'symbol', 'trade_side']]
            df_buy = df[df['trade_side'] == 1]
            if len(df_buy) == 0:
                print('当日无信号')
                return []
            else:
                # print('当日买入信号：%s' % df_buy.symbol.tolist())
                return df_buy.symbol.tolist()


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


def read_removed_st_mkt(positon='file'):
    '''读取风控需要被删除的股票的数据'''
    # positon = 'test'
    # 剔除掉的数据
    if positon == 'file':
        remove_data = []
        # now=time.strftime('%Y_%m_%d', time.localtime(time.time()))
        today = datetime.datetime.today()
        with open("c:\\e\\data\\qe\\backtest\\stock_blacklist", 'r', encoding='utf_8') as f:
            data = f.readlines()
            for line in data:
                if line and line.startswith('remove'):
                    info = line.strip().split('=')
                    removes = eval(info[1])
                    for key, values in removes.items():
                        begin_day = datetime.datetime.strptime(values[0], '%Y-%m-%d')
                        end_day = values[1]
                        if begin_day <= today and end_day:
                            end_date = datetime.datetime.strptime(values[1], '%Y-%m-%d')
                            if today <= end_date:
                                remove_data.append(key)
                        if begin_day <= today and (not end_day):
                            remove_data.append(key)
                if line and line.startswith('clean_all'):
                    info = line.strip().split('=')
                    clean_all = info[1]
        return remove_data, clean_all
    else:
        pass


def handle_buy(context, data):
    """
        开盘买入
    """
    Log = context.Log
    today = context.get_datetime().strftime('%Y-%m-%d')
    if context.get_datetime().strftime('%H:%M:%S') >= '14:56:00':
        today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # if not param.is_backtest:
    #	 today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    now_time = datetime.datetime.strptime((context.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')
    print(now_time)
    open_curent_time = datetime.datetime.strptime((today + ' 09:31:00'), '%Y-%m-%d %H:%M:%S')
    # open_curent_time = datetime.datetime.strptime((today + ' 14:00:00'), '%Y-%m-%d %H:%M:%S')
    Log.info('开盘买入,获取handle当前时间为：' + str(now_time))

    buy_info = context.buy_info
    # 如果没有要买入的股票则直接返回
    if not buy_info:
        Log.info('当前bar没有要买入的股票')
        return

    ''' 下面是测试设置 开盘就买入 考虑成交量'''
    deadline_time = datetime.datetime.strptime((today + ' 11:00:00'), '%Y-%m-%d %H:%M:%S')
    # deadline_time = datetime.datetime.strptime((today + ' 14:30:00'), '%Y-%m-%d %H:%M:%S')
    if open_curent_time <= now_time < deadline_time:
        # 循环判断连续阳线，如果成立就买入
        for market, hold in buy_info.items():
            print(hold)
            # 判断该只股票是否可以买入
            if data.can_trade(context.symbol(market)):

                # 判断当前bar的成交量限制
                if param.volume_limit > 0:
                    bar_limit = int(
                        float(context.current([market], ['volume'])) * 100 * param.volume_limit) // 100 * 100
                    print('volume: %s  limit: %s' % (context.current([market], ['volume']), bar_limit))

                    if bar_limit >= hold['buy_num']:
                        new_buy_num = hold['buy_num']
                        hold['buy_num'] = hold['buy_num'] - new_buy_num
                    else:
                        new_buy_num = bar_limit
                        hold['buy_num'] = hold['buy_num'] - new_buy_num
                else:
                    new_buy_num = hold['buy_num']

                # 执行买入下单
                context.order(context.symbol(market), new_buy_num)
                Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(new_buy_num))

        deepcopy_buy_info = deepcopy(buy_info)
        for market, hold in deepcopy_buy_info.items():
            if hold['buy_num'] == 0:
                # 该只股票已买入,从context中清理掉
                del context.buy_info[market]
        return
    elif now_time >= deadline_time:
        # 判断是否没有完全成交，如果没有，则强制买入
        if buy_info:
            for market, hold in buy_info.items():
                # 判断该只股票是否可以买入
                if data.can_trade(context.symbol(market)):
                    # 执行买入下单
                    context.order(context.symbol(market), hold['buy_num'])
                    Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
            context.buy_info = {}
        return
    return
    ''' 上面是测试设置 '''


def handle_sell(context, data):
    """
        新的卖出逻辑
    """
    Log = context.Log
    today = context.get_datetime().strftime('%Y-%m-%d')
    now_time = datetime.datetime.strptime((context.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')
    start_sell_time = datetime.datetime.strptime((today + ' 14:00:00'), '%Y-%m-%d %H:%M:%S')
    end_sell_time = datetime.datetime.strptime((today + ' 14:55:00'), '%Y-%m-%d %H:%M:%S')
    Log.info('收盘卖出,获取handle当前时间为：' + str(now_time))
    sell_markets = context.sell_markets

    # 如果没有要卖出的股票则直接返回
    if not sell_markets:
        Log.info('当前bar没有要卖出的股票')
        return
    if now_time == start_sell_time:
        global negative_k_count
        negative_k_count = {}
        for market, hold in sell_markets.items():
            negative_k_count[market] = 0

    ''' 下面是测试设置 收盘直接卖出'''
    start_sell_time = datetime.datetime.strptime((today + ' 14:30:00'), '%Y-%m-%d %H:%M:%S')
    if end_sell_time > now_time >= start_sell_time:
        # 循环判断连续阴线，如果成立就卖出
        for market, hold in sell_markets.items():

            # 判断当前bar的成交量限制
            if param.volume_limit > 0:
                bar_limit = int(float(context.current([market], ['volume'])) * 100 * param.volume_limit) // 100 * 100
                if bar_limit >= hold['amount']:
                    new_sell_num = hold['amount']
                    hold['amount'] = hold['amount'] - new_sell_num
                else:
                    new_sell_num = bar_limit
                    hold['amount'] = hold['amount'] - new_sell_num
            else:
                new_sell_num = hold['amount']

            # 执行卖出下单
            context.order(context.symbol(market), -new_sell_num)
            Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(new_sell_num))

        deepcopy_sell_markets = deepcopy(sell_markets)
        for market, hold in deepcopy_sell_markets.items():
            if hold['amount'] == 0:
                # 该只股票已买入,从context中清理掉
                del context.sell_markets[market]

        return
    elif now_time >= end_sell_time:
        # 判断是否没有完全成交，如果没有，则强制卖出
        if sell_markets:
            for market, hold in sell_markets.items():
                # 执行卖出下单
                context.order(context.symbol(market), -hold['amount'])
                Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(hold['amount']))
            context.sell_markets = {}
        return
    return
    ''' 上面是测试设置 '''


def handle_data(context, data):
    handle_buy(context=context, data=data)
    # 新的卖出逻辑
    handle_sell(context=context, data=data)


def get_portfolio(ContextInfo=None):
    '''
    获取账户总信息
    :param self:
    :return:
    '''

    print("查询账户总信息开始")
    # 0. 准备数据
    # 资金账号
    accountid = param.accountid
    # 账户类型，默认股票
    strAccountType = 'STOCK'

    # 1. 查询资金账户信息, 调用迅投 3.2.4.2(3) 函数
    strDatatype = 'ACCOUNT'
    try:
        account_info_list = get_trade_detail_data(accountid, strAccountType, strDatatype)
        print('get position', type(get_trade_detail_data))
    except:
        import traceback
        traceback.print_exc()
        print('error')
        import sys
        sys.exit('')
    # 2. 构造返回结果
    account_total_info_dict = {}
    for item in account_info_list:
        # 账户现金 float
        account_total_info_dict['cash'] = item.m_dAvailable
        # 交易账户对象统计开始时间 datetime
        account_total_info_dict['start_date'] = datetime.datetime.strptime(param.start_date, "%Y-%m-%d %H:%M:%S")
        # 交易账户对象统计结束时间 datetime
        account_total_info_dict['end_date'] = datetime.datetime.strptime(param.end_date, "%Y-%m-%d %H:%M:%S")
        # 交易账户初始金额 float
        account_total_info_dict['starting_cash'] = param.capital_base
        # 账户总价值（包括持仓市值+现金） float
        account_total_info_dict['portfolio_value'] = item.m_dBalance
        # 持仓市值 float
        account_total_info_dict['positions_value'] = item.m_dStockValue
        # 持仓 dictionary
        account_total_info_dict['positions'] = ContextInfo.get_positions()
        # 持仓风险暴露 float
        account_total_info_dict['positions_exposure'] = ''
        # 账户买卖所消耗的净资产(手续费)，为正时代表花费 float
        account_total_info_dict['capital_used'] = item.m_dCommission
        # 持仓收益 float
        account_total_info_dict['pnl'] = item.m_dPositionProfit
        # 账户累计收益，比如10%返回的是0.1  float
        account_total_info_dict['returns'] = "%.2f" % (item.m_dPositionProfit / param.capital_base)
    print('portfolio_value:%s' % account_total_info_dict['portfolio_value'])
    print('positions_value:%s' % account_total_info_dict['positions_value'])
    print('cash:%s' % account_total_info_dict['cash'])
    # 3. 按要求格式返回查询结果
    print("查询账户总信息完成")
    return account_total_info_dict


def get_positions(ContextInfo=None):
    '''
    查询持仓信息
    :return:
    '''
    print("查询持仓信息开始")
    # 0. 准备数据
    # 资金账号
    accountid = param.accountid
    # 账户类型，默认股票
    strAccountType = 'STOCK'
    # 1. 查询持仓信息, 调用迅投 3.2.4.2(3) 函数
    strDatatype = 'POSITION'
    position_info_list = get_trade_detail_data(accountid, strAccountType, strDatatype, ContextInfo.strategy_name)
    # 2. 构造返回结果
    position_dict = {}
    for item in position_info_list:
        position_target_dict = {}
        position_target_dict['amount'] = item.m_nVolume
        position_target_dict['cost_basis'] = item.m_dOpenPrice
        position_target_dict['last_sale_price'] = item.m_dSettlementPrice
        position_target_dict['sid'] = item.m_strInstrumentID
        position_target_dict['last_sale_date'] = datetime.date.today()
        position_target_dict['asset'] = None
        position_dict[item.m_strInstrumentID] = position_target_dict
    # 3. 按要求格式返回查询结果
    print("查询持仓信息完成")
    return position_dict


def can_trade(ContextInfo, stockcode):
    '''
    该股票是否可以交易，返回值类型是布尔型
    :param stockcode: String 股票代码
    :return: number
    '''
    # 使用迅投 3.2.3(1) 函数进行设置
    number = ContextInfo.get_last_volume(stockcode)
    if number <= 0:
        return False
    else:
        return True


def current(ContextInfo, assets, fields):
    '''
    获取当天股票数据(从开盘到当前时刻的订阅股票列表的各种数据)
    备注：到当前时刻，所以end_time不设置
    :return:
    '''
    '''
    fields：字段列表：
      'open'：开  'high'：高  'low'：低  'close'：收
      'volume'：成交量  'amount'：成交额  'settle'：结算价
    stock_code: list 股票代码列表
    start_time：默认参数，开始时间，格式 '20171209' 或 '20171209010101'
    end_time：默认参数，结束时间，格式 '20171209' 或 '20171209010101'
    skip_paused：默认参数，可选值：
      true：如果是停牌股，会自动填充未停牌前的价格作为停牌日的价格
      False：停牌数据为nan
    period：默认参数，周期类型：
      'tick'：分笔线  '1d'：日线  '1m'：1分钟线  '3m'：3分钟线  '5m'：5分钟线
      '15m'：15分钟线  '30m'：30分钟线  '1h'：小时线  '1w'：周线  '1mon'：月线
      '1q'：季线  '1hy'：半年线  '1y'：年线
    dividend_type：默认参数，缺省值为 'none'，除复权，可选值：
      'none'：不复权  'front'：向前复权  'back'：向后复权  'front_ratio'：等比向前复权  'back_ratio'：等比向后复权
    count： = -1 ，让start_time 和 end_time生效
    '''
    # 获取股票池中的股票，使用迅投 3.2.2(7) 函数
    stock_code = assets
    # 获取周期
    period = ContextInfo.period
    dividend_type = 'none'
    # 使用迅投 3.2.3(17) 函数进行设置
    df = ContextInfo.get_market_data(fields, stock_code, skip_paused=True, period=period, dividend_type=dividend_type,
                                     count=2)
    df = df[fields[0]]
    df = df[0]
    return df


def symbol(symbol_str):
    '''
    通过股票代码生成股票对象列表 list
    :return:
    '''
    list = []
    list.append(symbol_str)
    return symbol_str


def get_order(ContextInfo, order=None):
    '''
    获取当天已成交的订单
    :param order:
    :return:
    '''
    print("获取已成交的订单操作开始")
    # 0. 准备数据
    # 资金账号
    accountid = ContextInfo.accID
    print(accountid)
    # 账户类型，默认股票
    strAccountType = 'STOCK'

    # 1. 获取交易明细中成交对象列表，并将 m_strOrderSysID 保存为成交order_id列表， 调用迅投 3.2.4.2(3) 函数
    strDatatype_deal = 'DEAL'
    deal_list = get_trade_detail_data(accountid, strAccountType, strDatatype_deal)
    # 2. 构造返回结果
    deal_dict = {}
    for item in deal_list:
        deal_target_dict = {}
        deal_target_dict['date'] = item.m_strTradeDate + item.m_strTradeTime  # 成交时间
        deal_target_dict['amount'] = item.m_nVolume  # 成交量
        deal_target_dict['price'] = item.m_dPrice  # 成交均价
        deal_target_dict['comssion'] = item.m_dComssion  # 手续费
        deal_target_dict['total'] = item.m_dPrice * item.m_nVolume + item.m_dComssion  # 总花费： 成交均价 * 成交量 + 手续费
        deal_dict[item.m_strInstrumentID] = deal_target_dict
    # 3. 按要求格式返回查询结果
    print("获取已成交的订单操作完成")
    return deal_dict


def order(ContextInfo, asset, amount, position_effect=None, limit_price=None, stop_price=None, style=None):
    '''
    生成订单，即下单操作
    :return:
    '''

    # opType, orderType, orderCode, prType, volume, ContextInfo
    '''
      实例：passorder(23,1202, 'testS', '000001.SZ', 5, -1, 50000, ContextInfo)，意思就是对账号组 testS 里的
      所有账号都以最新价开仓买入 50000 元市值的 000001.SZ 平安银行
    --opType：number，可选值：
      23：股票买入，或沪港通、深港通股票买入
      24：股票卖出，或沪港通、深港通股票卖出
    --orderType，下单方式，可选值：
      1101：单股、单账号、普通、股/手方式下单
      1113：单股、单账号、总资产、比例 [0 ~ 1] 方式下单
      1123：单股、单账号、可用、比例[0 ~ 1]方式下单
    --accountID，资金账号：下单的账号ID（可多个），目前咱们默认是 基础资金账号: '26769922'
    --orderCode，下单代码（对于股票就是股票代码）
    --prType，下单选价类型，可选值：
      0：卖5价 1：卖4价 2：卖3价 3：卖2价 4：卖1价 5：最新价
      6：买1价 7：买2价 8：买3价 9：买4价 10：买5价
    --modelprice，模型下单价格，因为咱们使用的prType不是模型类型，所以该字段可以任意填写，默认为 -1
    --volume，下单数量（股、手 / 元 / %） 期货交易单位是手，股票交易单位是股
      根据 orderType 值最后一位确定 volume 的单位：
      单股下单时：1：股 / 手  2：金额（元） 3：比例（%）
    --quickTrade，int，设定是否立即触发下单，可选值：
      0：否
      1：是 --咱们目前是 1
    *注：passorder是对最后一根 K 线完全走完后生成的模型信号在下一根 K 线的第一个tick数据来时触发下单交易；
      采用quickTrade参数设置为1时，只要策略模型中调用到passorder交易函数就触发下单交易。
    '''

    print("生成订单：")
    print("股票代码为：" + asset + " ;下单数量为：" + str(amount))

    # 根据amount，控制买入或卖出操作
    if amount > 0:
        opType = 23
    elif amount < 0:
        opType = 24
    else:
        return

    orderType = 1101
    orderCode = asset
    accountid = param.accountid
    prType = 5
    modelprice = -1
    volume = amount
    quickTrade = 1
    # strategyName = ContextInfo.strategy_name
    # 使用迅投 3.2.4.2(1) 函数进order行下单
    print(orderCode)
    print(type(passorder))
    passorder(opType, orderType, accountid, orderCode, prType, modelprice, abs(volume), ContextInfo.strategy_name,
              quickTrade, ContextInfo, )


def get_stocklist(context):
    # 获取订阅股票池
    # instruments = ['002687.SZ', '300344.SZ', '601766.SH', '601601.SH', '300491.SZ', '601186.SH', '600028.SH', '300169.SZ', '300690.SZ', '300242.SZ', '300374.SZ', '300593.SZ', '600463.SH', '300648.SZ', '002858.SZ', '601155.SH', '300264.SZ', '300687.SZ', '603288.SH', '600837.SH', '603667.SH', '300345.SZ', '603002.SH', '300720.SZ', '603908.SH', '000668.SZ', '002494.SZ', '000858.SZ', '300654.SZ', '601166.SH', '300165.SZ', '002931.SZ', '603696.SH', '002558.SZ', '002810.SZ', '300240.SZ', '603966.SH', '603917.SH', '600031.SH', '600731.SH', '300750.SZ', '002825.SZ', '002136.SZ', '002066.SZ', '002580.SZ', '600802.SH', '600080.SH', '600513.SH', '600066.SH', '600051.SH', '300161.SZ', '600113.SH', '300112.SZ', '300003.SZ', '000023.SZ', '000705.SZ', '600192.SH', '300405.SZ', '300542.SZ', '002210.SZ', '601009.SH', '300645.SZ', '603477.SH', '300543.SZ', '300589.SZ', '600319.SH', '300587.SZ', '603086.SH', '600883.SH', '000776.SZ', '600893.SH', '600048.SH', '300407.SZ', '002202.SZ', '000707.SZ', '300163.SZ', '002456.SZ', '600276.SH', '600539.SH', '300619.SZ', '603136.SH', '300688.SZ', '603321.SH', '300640.SZ', '603028.SH', '600398.SH', '300421.SZ', '002676.SZ', '600156.SH', '300497.SZ', '600311.SH', '002760.SZ', '002856.SZ', '601238.SH', '002509.SZ', '600152.SH', '300736.SZ', '002017.SZ', '300471.SZ', '300086.SZ', '300004.SZ', '002259.SZ', '300260.SZ', '300693.SZ', '300637.SZ', '603813.SH', '002175.SZ', '300615.SZ', '002501.SZ', '002529.SZ', '000637.SZ', '300023.SZ', '002052.SZ', '002723.SZ', '300534.SZ', '603615.SH', '002024.SZ', '600029.SH', '300551.SZ', '600359.SH', '600735.SH', '000333.SZ', '300335.SZ', '300046.SZ', '000783.SZ', '300312.SZ', '601818.SH', '300277.SZ', '300631.SZ', '600130.SH', '002524.SZ', '600660.SH', '603041.SH', '300518.SZ', '603009.SH', '002417.SZ', '603800.SH', '600599.SH', '300105.SZ', '300585.SZ', '600078.SH', '600405.SH', '000002.SZ', '300228.SZ', '600397.SH', '000576.SZ', '603895.SH', '601933.SH', '000716.SZ', '300124.SZ', '002134.SZ', '002432.SZ', '300592.SZ', '600365.SH', '002870.SZ', '002575.SZ', '300626.SZ', '601688.SH', '300556.SZ', '300126.SZ', '300278.SZ', '002729.SZ', '600556.SH', '002865.SZ', '300449.SZ', '601088.SH', '300084.SZ', '600830.SH', '600237.SH', '300703.SZ', '002072.SZ', '002896.SZ', '600727.SH', '300356.SZ', '600354.SH', '300122.SZ', '603388.SH', '600462.SH', '300072.SZ', '000419.SZ', '300431.SZ', '600778.SH', '600309.SH', '300160.SZ', '002289.SZ', '002660.SZ', '000509.SZ', '002272.SZ', '300013.SZ', '300107.SZ', '000893.SZ', '000790.SZ', '300076.SZ', '603268.SH', '600538.SH', '603822.SH', '600900.SH', '000868.SZ', '300280.SZ', '000611.SZ', '600222.SH', '002765.SZ', '300472.SZ', '600796.SH', '600030.SH', '603726.SH', '002354.SZ', '600520.SH', '002142.SZ', '603729.SH', '600436.SH', '002248.SZ', '603809.SH', '601169.SH', '002875.SZ', '002714.SZ', '603286.SH', '300030.SZ', '600016.SH', '601668.SH', '300444.SZ', '300157.SZ', '603903.SH', '300498.SZ', '600011.SH', '002196.SZ', '603006.SH', '600600.SH', '300647.SZ', '603396.SH', '002150.SZ', '603580.SH', '603289.SH', '000001.SZ', '300453.SZ', '603488.SH', '600980.SH', '601600.SH', '601988.SH', '002667.SZ', '603335.SH', '603937.SH', '601877.SH', '600188.SH', '002054.SZ', '600958.SH', '002921.SZ', '600191.SH', '000731.SZ', '002536.SZ', '603617.SH', '002021.SZ', '000856.SZ', '300089.SZ', '300555.SZ', '603535.SH', '300517.SZ', '601628.SH', '300404.SZ', '300192.SZ', '600706.SH', '300268.SZ', '300489.SZ', '000963.SZ', '002873.SZ', '600692.SH', '600115.SH', '600470.SH', '600766.SH', '600889.SH', '600838.SH', '002278.SZ', '300397.SZ', '002576.SZ', '300535.SZ', '300211.SZ', '001979.SZ', '600689.SH', '002044.SZ', '603655.SH', '000702.SZ', '600833.SH', '000017.SZ', '002213.SZ', '300707.SZ', '300162.SZ', '002606.SZ', '600532.SH', '300283.SZ', '300097.SZ', '601939.SH', '603208.SH', '603183.SH', '601989.SH', '002198.SZ', '300306.SZ', '300659.SZ', '300246.SZ', '300490.SZ', '600867.SH', '300214.SZ', '600240.SH', '002843.SZ', '000835.SZ', '600186.SH', '600361.SH', '603050.SH', '603380.SH', '600025.SH', '002849.SZ', '000955.SZ', '002809.SZ', '600000.SH', '300179.SZ', '300234.SZ', '300584.SZ', '600165.SH', '002356.SZ', '000018.SZ', '300380.SZ', '600637.SH', '300635.SZ', '300563.SZ', '600768.SH', '300093.SZ', '300462.SZ', '603022.SH', '300270.SZ', '603059.SH', '002027.SZ', '600821.SH', '002077.SZ', '601919.SH', '601998.SH', '600015.SH', '601012.SH', '600112.SH', '000721.SZ', '603320.SH', '002872.SZ', '600095.SH', '300025.SZ', '300509.SZ', '603829.SH', '002624.SZ', '600678.SH', '603506.SH', '002917.SZ', '603110.SH', '603722.SH', '002748.SZ', '002236.SZ', '300042.SZ', '300717.SZ', '600213.SH', '002629.SZ', '002836.SZ', '300295.SZ', '000691.SZ', '600898.SH', '600209.SH', '300680.SZ', '600448.SH', '300371.SZ', '000785.SZ', '000338.SZ', '002045.SZ', '300029.SZ', '002584.SZ', '600235.SH', '300636.SZ', '300818.SZ', '002103.SZ', '603658.SH', '603859.SH', '300478.SZ', '300095.SZ', '002112.SZ', '300139.SZ', '300598.SZ', '600747.SH', '000506.SZ', '600540.SH', '002427.SZ', '002943.SZ', '002862.SZ', '300632.SZ', '300090.SZ', '002778.SZ', '603703.SH', '600297.SH', '002231.SZ', '603037.SH', '300164.SZ', '600128.SH', '000014.SZ', '002919.SZ', '300402.SZ', '600741.SH', '300722.SZ', '300071.SZ', '300650.SZ', '300218.SZ', '600543.SH', '000755.SZ', '002908.SZ', '603200.SH', '600592.SH', '603767.SH', '600438.SH', '002883.SZ', '002817.SZ', '600212.SH', '002890.SZ', '000812.SZ', '603683.SH', '300148.SZ', '002755.SZ', '002347.SZ', '600809.SH', '000816.SZ', '300655.SZ', '300103.SZ', '002205.SZ', '002808.SZ', '002076.SZ', '600036.SH', '600560.SH', '603536.SH', '002591.SZ', '002758.SZ', '601225.SH', '002105.SZ', '600519.SH', '300354.SZ', '601328.SH', '600810.SH', '600018.SH', '603880.SH', '002164.SZ', '000554.SZ', '603269.SH', '603127.SH', '300210.SZ', '300056.SZ', '300362.SZ', '002622.SZ', '600892.SH', '002696.SZ', '600249.SH', '600721.SH', '300074.SZ', '002220.SZ', '600107.SH', '300290.SZ', '600119.SH', '300235.SZ', '600099.SH', '300330.SZ', '002321.SZ', '300522.SZ', '300729.SZ', '002311.SZ', '300711.SZ', '600019.SH', '603031.SH', '002357.SZ', '600671.SH', '002120.SZ', '603721.SH', '600159.SH', '000737.SZ', '603787.SH', '002422.SZ', '601211.SH', '000609.SZ', '600009.SH', '603066.SH', '601336.SH', '002089.SZ', '601006.SH', '603500.SH', '300430.SZ', '002360.SZ', '000571.SZ', '601800.SH', '603330.SH', '603389.SH', '300501.SZ', '002694.SZ', '600104.SH', '300721.SZ', '600355.SH', '300710.SZ', '300225.SZ', '603160.SH', '600085.SH', '002296.SZ', '600203.SH', '002189.SZ', '300070.SZ', '300049.SZ', '300606.SZ', '603879.SH', '603819.SH', '600122.SH', '300015.SZ', '002026.SZ', '603709.SH', '300492.SZ', '603398.SH', '002805.SZ', '600290.SH', '600506.SH', '002316.SZ', '603499.SH', '603993.SH', '002122.SZ', '300743.SZ', '300022.SZ', '600346.SH', '603178.SH', '000596.SZ', '000538.SZ', '600328.SH', '601360.SH', '002835.SZ', '002731.SZ', '300254.SZ', '603161.SH', '000595.SZ', '300032.SZ', '300475.SZ', '600371.SH', '601888.SH', '300419.SZ', '000890.SZ', '002869.SZ', '300205.SZ', '600585.SH', '300712.SZ', '603757.SH', '603970.SH', '300515.SZ', '002886.SZ', '603139.SH', '002209.SZ', '600634.SH', '002860.SZ', '300730.SZ', '300581.SZ', '002803.SZ', '603266.SH', '000159.SZ', '300505.SZ', '603778.SH', '300281.SZ', '603045.SH', '603922.SH', '300092.SZ', '002740.SZ', '601398.SH', '300526.SZ', '603088.SH', '002455.SZ', '002012.SZ', '002569.SZ', '300697.SZ', '300540.SZ', '600615.SH', '600257.SH', '600196.SH', '300069.SZ', '002715.SZ', '002415.SZ', '600887.SH', '002816.SZ', '002762.SZ', '300167.SZ', '000708.SZ', '600677.SH', '300106.SZ', '002304.SZ', '601857.SH', '603227.SH', '300716.SZ', '300239.SZ', '300445.SZ', '300275.SZ', '300554.SZ', '000651.SZ', '601607.SH', '601318.SH', '603165.SH', '603725.SH', '002493.SZ', '000909.SZ', '600690.SH', '600666.SH', '603196.SH', '000613.SZ', '000786.SZ', '002871.SZ', '600455.SH', '002352.SZ', '601288.SH', '600746.SH', '601111.SH', '300733.SZ', '300152.SZ', '300334.SZ', '603633.SH', '603042.SH', '300375.SZ', '002442.SZ', '002711.SZ', '002594.SZ', '300605.SZ', '600050.SH', '600139.SH', '600606.SH', '300668.SZ', '300700.SZ', '600281.SH', '300063.SZ', '300537.SZ', '300062.SZ', '300536.SZ', '300446.SZ', '300452.SZ', '300175.SZ', '603776.SH', '002492.SZ', '300629.SZ', '002899.SZ', '300622.SZ']

    if not param.is_backtest:
        if param.start_date[:10] > '2020-09-21':
            instruments = planedorders('2020-09-21', param.end_date[:10], param.strategy_map[context.strategy_name])
        else:
            instruments = planedorders(param.start_date[:10], param.end_date[:10],
                                       param.strategy_map[context.strategy_name])
        # instruments = planedorders(param.start_date[:10], param.end_date[:10], param.strategy_map[context.strategy_name])
        instruments = list(set(instruments))
        return instruments
    else:
        if param.start_date[:10] > '2020-09-21':
            instruments = planedorders('2020-09-21', param.end_date[:10], param.strategy_map[context.strategy_name])
        else:
            instruments = planedorders(param.start_date[:10], param.end_date[:10],
                                       param.strategy_map[context.strategy_name])
        instruments = list(set(instruments))
        return instruments


def current_dt(ContextInfo):
    '''
    返回k线时间
    :param ContextInfo:
    :return:
    '''
    timetag = ContextInfo.get_bar_timetag(ContextInfo.barpos)
    now = timtag_to_datetime(timetag, '%Y-%m-%d %H:%M:%S')

    # print('========', now)
    return datetime.datetime.strptime(now, '%Y-%m-%d %H:%M:%S')


def timtag_to_datetime(timetag, format):
    import time
    timetag = timetag / 1000
    time_local = time.localtime(timetag)
    return time.strftime(format, time_local)


def trans(qmt_df):
    '''
    将迅投数据与中信数据格式进行转换（3D->3D）
    :param qmt_df:
    :return:
    '''
    qmt_df = qmt_df.transpose((2, 1, 0))
    tmp = qmt_df.major_axis
    tmp_l = [datetime.strptime(i, '%Y%m%d') for i in tmp]
    qmt_df.major_axis = tmp_l
    return qmt_df


def panel_wf2_df(the_panel):
    '''
    3D -> 2D panel -> dataframe
    :param the_panel:
    :return:
    '''
    the_panel = the_panel.transpose((2, 1, 0))
    b = the_panel.to_frame()
    d = b.reset_index()
    d.rename(columns={'major': 'date', 'minor': 'instrument'}, inplace=True)
    return d
