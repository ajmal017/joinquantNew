# @Time    : 2020/6/29 11:15
# @Author  : XuWenFu


import math
import time
import datetime
import numpy as np
from data_function import *
from copy import deepcopy


'''
股票代码交易部分

'''

# 连续几个阳线（包括当前这根）
positive_k_count = {}

# 连续几个阴线（包括当前这根）
negative_k_count = {}

# 从开盘(不包括第一根)到现在（包括当前这根）的最低价
lowest_price = {}

# 从下午2点开始(不包括第一根)到现在（包括当前这根）的最高价
highest_price = {}


def read_minute_history(start_date='2018-01-01', end_date='2020-06-25', instruments=None, traget_fields=None,
                        flag='bar1m', market='AShare', positon='xuntou', context=None):
    # 迅投分钟级别数据读取

    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    assets = instruments
    period = '1m'
    # 构建datetime对象
    start_date = datetime.datetime.strptime((start_date + ' 09:30:00'), '%Y-%m-%d %H:%M:%S')
    end_date = datetime.datetime.strptime((end_date + ' 15:00:00'), '%Y-%m-%d %H:%M:%S')

    # 计算出str
    start_time = start_date.strftime('%Y%m%d%H%M%S')
    end_time = end_date.strftime('%Y%m%d%H%M%S')

    dividend_type = 'none'

    # 使用迅投 3.2.3(17) 函数进行设置
    df = context.get_market_data(fields, assets, start_time, end_time,
                                 skip_paused=True, period=period, dividend_type=dividend_type, count=-1)

    df = trans(df)

    if df.shape:
        df['turover'] = df['turover'].fillna(method='pad')
        df['close'] = df['close'].fillna(method='pad')
        df['high'] = df['high'].fillna(method='pad')
        df['low'] = df['low'].fillna(method='pad')
        df['volume'] = df['volume'].fillna(0)
    return df


def read_removed_st_mkt(positon='file'):
    '''读取风控需要被删除的股票的数据'''
    # positon = 'test'
    # 剔除掉的数据
    if positon == 'file':
        remove_data = []
        # now=time.strftime('%Y_%m_%d', time.localtime(time.time()))
        today = datetime.datetime.today()
        with open("c:\\e\\data\\qe\\backtest\\stock_blacklist", 'r', encoding = 'utf_8') as f:
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


def trade_risk_manage(context, market):
    """
        交易时的风险管理
    """
    """
    Log = context.Log
    mkt_sts,clean_all = read_removed_st_mkt()
    if market in mkt_sts:
        Log.info('此单被风险排除掉，不交易' + str(market))
        return False
    if clean_all==1:
        return False
    else:
        return True
    """
    return True


def handle_buy(context, data):
    """
        开盘买入
    """
    Log = context.Log
    today = context.get_datetime().strftime('%Y-%m-%d')
    if context.get_datetime().strftime('%H:%M:%S') >= '14:56:00':
        today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # if not param.is_backtest:
    #     today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    now_time = datetime.datetime.strptime((context.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')
    print(now_time)
    # open_curent_time = datetime.datetime.strptime((today + ' 09:31:00'), '%Y-%m-%d %H:%M:%S')
    open_curent_time = datetime.datetime.strptime((today + ' 11:20:00'), '%Y-%m-%d %H:%M:%S')
    Log.info('开盘买入,获取handle当前时间为：' + str(now_time))

    # # 计算要买入的数量（整100的倍数）
    # if now_time == open_curent_time:
    #     # 全局变量清零
    #     global positive_k_count
    #     global lowest_price
    #     global highest_price
    #     # global buy_vol
    #     # buy_vol = {}
    #     positive_k_count = {}
    #     lowest_price = {}
    #     highest_price = {}
    #     buy_info = deepcopy(context.buy_info)
    #     new_buy_info = {}
    #     # 开始计算
    #     for market, hold in buy_info.items():
    #         # df_cur = data.current([context.symbol(market)], ['close', 'open', 'high', 'low', 'volume', 'price'])
    #         # close_cul = df_cur.loc[context.symbol(market)]['close']
    #         print(market)
    #         if market in ['300090.SZ', '600240.SH', '300362.SZ', '600069.SH', '300028.SZ', '300156.SZ']:
    #             continue
    #         close_cul = float(context.current([market], ['close']))
    #         context.current([market], ['close'])
    #         print(close_cul)
    #
    #         buy_num = int((hold['buy_cash'] // (100 * close_cul)) * 100)
    #         market_info = {
    #                         'generate_time': hold['generate_time'],
    #                         'buy_num': buy_num,
    #                         'buy_cash': hold['buy_cash']
    #                         #'s_dq_close': hold['s_dq_close']
    #                       }
    #         # buy_vol[market] = buy_num
    #         # print(buy_vol[market])
    #         positive_k_count[market] = 0
    #         new_buy_info[market] = market_info
    #         lowest_price[market] = close_cul
    #     # 替换掉buy_info
    #     context.buy_info = new_buy_info

    buy_info = context.buy_info
    # 如果没有要买入的股票则直接返回
    if not buy_info:
        Log.info('当前bar没有要买入的股票')
        return

    ''' 下面是测试设置 开盘就买入 '''
    # if now_time == open_curent_time:
    #     # 循环判断连续阳线，如果成立就买入
    #     for market, hold in buy_info.items():
    #         # 判断该只股票是否可以买入
    #         if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #             # 执行买入下单
    #             context.order(context.symbol(market), hold['buy_num'])
    #             Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #
    #         # 该只股票已买入,从context中清理掉
    #         del context.buy_info[market]
    #     return
    # else:
    #     return
    ''' 上面是测试设置 '''
    ''' 下面是测试设置 开盘就买入 考虑成交量'''
    # deadline_time = datetime.datetime.strptime((today + ' 11:00:00'), '%Y-%m-%d %H:%M:%S')
    deadline_time = datetime.datetime.strptime((today + ' 11:30:00'), '%Y-%m-%d %H:%M:%S')
    if open_curent_time <= now_time < deadline_time:
        # 循环判断连续阳线，如果成立就买入
        for market, hold in buy_info.items():
            print(hold)
            # 判断该只股票是否可以买入
            if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):

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
                if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
                    # 执行买入下单
                    context.order(context.symbol(market), hold['buy_num'])
                    Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
            context.buy_info = {}
        return
    return
    ''' 上面是测试设置 '''
    ''' 下面是测试设置 有效阳线买入 逻辑3'''
    # 判断一个时间范围：9：30 - 10：59，则执行判断连续阳线的逻辑; 11：00 执行强制全部买入的逻辑
    # deadline_time = datetime.datetime.strptime((today + ' 11:00:00'), '%Y-%m-%d %H:%M:%S')
    # if open_curent_time <= now_time < deadline_time:
    #     # 循环判断连续阳线，如果成立就买入
    #     for market, hold in buy_info.items():
    #         # 判断该只股票是否可以买入
    #         if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #             # 全局变量
    #             # global positive_k_count
    #             # 判断是否为连续n根阳线，如果是则执行买入操作
    #             if positive_k_count[market] < 1:
    #
    #                 open_cul = context.current([market], ['open'])
    #                 close_cul = context.current([market], ['close'])
    #                 print(open_cul, close_cul)
    #                 if close_cul / open_cul > 1:
    #                     positive_k_count[market] = positive_k_count[market] + 1
    #                 else:
    #                     positive_k_count[market] = 0
    #             if positive_k_count[market] >= 1:
    #                 # 判断当前bar的成交量限制
    #                 if param.volume_limit > 0:
    #                     bar_limit = int(
    #                         float(context.current([market], ['volume'])) * 100 * param.volume_limit) // 100 * 100
    #                     print('volume: %s  limit: %s' % (context.current([market], ['volume']), bar_limit))
    #
    #                     if bar_limit >= hold['buy_num']:
    #                         new_buy_num = hold['buy_num']
    #                         hold['buy_num'] = hold['buy_num'] - new_buy_num
    #                     else:
    #                         new_buy_num = bar_limit
    #                         hold['buy_num'] = hold['buy_num'] - new_buy_num
    #                 else:
    #                     new_buy_num = hold['buy_num']
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), new_buy_num)
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(new_buy_num))
    #
    #     deepcopy_buy_info = deepcopy(buy_info)
    #     for market, hold in deepcopy_buy_info.items():
    #         if hold['buy_num'] == 0:
    #             # 该只股票已买入,从context中清理掉s
    #             del context.buy_info[market]
    #
    #     return
    # # 11：00 执行强制全部买入的逻辑
    # elif now_time >= deadline_time:
    #     # 判断是否没有完全成交，如果没有，则强制买入
    #     if buy_info:
    #         for market, hold in buy_info.items():
    #             # 判断该只股票是否可以买入
    #             if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), hold['buy_num'])
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #
    #         context.buy_info = {}
    #     return
    # return
    ''' 下面是测试设置 有效阳线买入 逻辑7'''
    # 判断一个时间范围：9：30 - 10：59，则执行判断有效阳线的逻辑; 11：00 执行强制全部买入的逻辑
    # deadline_time = datetime.datetime.strptime((today + ' 11:00:00'), '%Y-%m-%d %H:%M:%S')
    # if open_curent_time <= now_time < deadline_time:
    #     # 循环判断连续阳线，如果成立就买入
    #     for market, hold in buy_info.items():
    #         # 判断该只股票是否可以买入
    #         if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #
    #
    #             open_cul = context.current([market], ['open'])
    #             close_cul = context.current([market], ['close'])
    #             print(open_cul, close_cul)
    #             if close_cul / open_cul > 1 + 0.001:
    #
    #                 # 判断当前bar的成交量限制
    #                 if param.volume_limit > 0:
    #                     bar_limit = int(
    #                         float(context.current([market], ['volume'])) * 100 * param.volume_limit) // 100 * 100
    #                     print('volume: %s  limit: %s' % (context.current([market], ['volume']), bar_limit))
    #
    #                     if bar_limit >= hold['buy_num']:
    #                         new_buy_num = hold['buy_num']
    #                         hold['buy_num'] = hold['buy_num'] - new_buy_num
    #                     else:
    #                         new_buy_num = bar_limit
    #                         hold['buy_num'] = hold['buy_num'] - new_buy_num
    #                 else:
    #                     new_buy_num = hold['buy_num']
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), new_buy_num)
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(new_buy_num))
    #
    #     deepcopy_buy_info = deepcopy(buy_info)
    #     for market, hold in deepcopy_buy_info.items():
    #         if hold['buy_num'] == 0:
    #             # 该只股票已买入,从context中清理掉s
    #             del context.buy_info[market]
    #
    #     return
    # # 11：00 执行强制全部买入的逻辑
    # elif now_time >= deadline_time:
    #     # 判断是否没有完全成交，如果没有，则强制买入
    #     if buy_info:
    #         for market, hold in buy_info.items():
    #             # 判断该只股票是否可以买入
    #             if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), hold['buy_num'])
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #
    #         context.buy_info = {}
    #     return
    # return
    ''' 上面是测试设置 '''

    ''' 下面是测试设置 比前一个交易日的收盘价低 0.98 就买入 '''
    # 比前一个交易日的收盘价低 0.98 就买入 ; 11：00 执行强制全部买入的逻辑
    # deadline_time = datetime.datetime.strptime((today + ' 11:00:00'), '%Y-%m-%d %H:%M:%S')
    # if now_time < deadline_time:
    #     for market, hold in buy_info.items():
    #         # 判断该只股票是否可以买入
    #         if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #             df_cur = data.current([context.symbol(market)], ['close', 'open', 'high', 'low', 'volume', 'price'])
    #             close_cul = df_cur.loc[context.symbol(market)]['close']
    #             if close_cul < hold['s_dq_close'] * 0.98:
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), hold['buy_num'])
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #                 # 该只股票已买入,从context中清理掉
    #                 del context.buy_info[market]
    #
    # # 11：00 执行强制全部买入的逻辑
    # elif now_time == deadline_time:
    #     # 判断是否没有完全成交，如果没有，则强制买入
    #     if buy_info:
    #         for market, hold in buy_info.items():
    #             # 判断该只股票是否可以买入
    #             if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), hold['buy_num'])
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #
    #         context.buy_info = {}
    #
    # return
    ''' 上面是测试设置 '''

    ''' 下面是测试设置 比今日最低价高千分之五 就买入 '''
    # 比今日最低价高千分之五 就买入 ; 11：00 执行强制全部买入的逻辑
    # deadline_time = datetime.datetime.strptime((today + ' 11:00:00'), '%Y-%m-%d %H:%M:%S')
    # if now_time < deadline_time:
    #     for market, hold in buy_info.items():
    #         # 判断该只股票是否可以买入
    #         if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #             # 全局变量
    #             global lowest_price
    #             df_cur = data.current([context.symbol(market)], ['close', 'open', 'high', 'low', 'volume', 'price'])
    #             close_cul = df_cur.loc[context.symbol(market)]['close']
    #
    #             if close_cul < lowest_price[market]:
    #                 lowest_price[market] = close_cul
    #
    #             if (close_cul - lowest_price[market])/lowest_price[market] > 0.005:
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), hold['buy_num'])
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #                 # 该只股票已买入,从context中清理掉
    #                 del context.buy_info[market]
    #
    # # 11：00 执行强制全部买入的逻辑
    # elif now_time == deadline_time:
    #     # 判断是否没有完全成交，如果没有，则强制买入
    #     if buy_info:
    #         for market, hold in buy_info.items():
    #             # 判断该只股票是否可以买入
    #             if trade_risk_manage(context=context, market=market) and data.can_trade(context.symbol(market)):
    #                 # 执行买入下单
    #                 context.order(context.symbol(market), hold['buy_num'])
    #                 Log.info("买入的股票代码是：" + str(market) + "买入的数量是：" + str(hold['buy_num']))
    #
    #         context.buy_info = {}
    #
    # return
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

    ''' 下面是测试设置 如果低于当日最高价的千分之五 则卖出'''
    # 判断一个时间范围：14：00 - 14：54，如果低于当日最高价的千分之五 则卖出; 14：55 执行强制全部卖出的逻辑
    # if start_sell_time <= now_time < end_sell_time:
    #     # 循环判断连续阴线，如果成立就卖出
    #     for market, hold in sell_markets.items():
    #         # 全局变量
    #         global highest_price
    #         # 更新连续阴线的全局变量
    #         df_cur = data.current([context.symbol(market)], ['close', 'open', 'high', 'low', 'volume', 'price'])
    #         close_cul = df_cur.loc[context.symbol(market)]['close']
    #
    #         if now_time == start_sell_time:
    #             highest_price[market] = close_cul
    #
    #         if close_cul > highest_price[market]:
    #             highest_price[market] = close_cul
    #
    #         if (highest_price[market] - close_cul) / highest_price[market] > 0.005:
    #             # 执行卖出下单
    #             context.order(context.symbol(market), -hold['amount'])
    #             Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(hold['amount']))
    #             # 该只股票已卖出,从context中清理掉
    #             del context.sell_markets[market]
    #
    # # 到 14：55 执行强制全部卖出的逻辑
    # elif now_time == end_sell_time:
    #     # 判断是否没有完全成交，如果没有，则强制卖出
    #     if sell_markets:
    #         for market, hold in sell_markets.items():
    #             # 执行卖出下单
    #             context.order(context.symbol(market), -hold['amount'])
    #             Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(hold['amount']))
    #         context.sell_markets = {}
    #
    # return
    ''' 上面是测试设置 '''

    ''' 下面是测试设置 有效阴线卖出 逻辑3'''

    # if start_sell_time <= now_time < end_sell_time:
    #     # 循环判断连续阴线，如果成立就卖出
    #     for market, hold in sell_markets.items():
    #         # 全局变量
    #         # global negative_k_count
    #         if negative_k_count[market] < 1:
    #             open_cul = context.current([market], ['open'])
    #             close_cul = context.current([market], ['close'])
    #             if close_cul/open_cul < 1:
    #                 negative_k_count[market] = negative_k_count[market] + 1
    #             else:
    #                 negative_k_count[market] = 0
    #         if negative_k_count[market] >= 1:
    #             if param.volume_limit > 0:
    #                 bar_limit = int(
    #                     float(context.current([market], ['volume'])) * 100 * param.volume_limit) // 100 * 100
    #                 if bar_limit >= hold['amount']:
    #                     new_sell_num = hold['amount']
    #                     hold['amount'] = hold['amount'] - new_sell_num
    #                 else:
    #                     new_sell_num = bar_limit
    #                     hold['amount'] = hold['amount'] - new_sell_num
    #             else:
    #                 new_sell_num = hold['amount']
    #
    #             # 执行卖出下单
    #             context.order(context.symbol(market), -new_sell_num)
    #             Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(new_sell_num))
    #     deepcopy_sell_markets = deepcopy(sell_markets)
    #     for market, hold in deepcopy_sell_markets.items():
    #         if hold['amount'] == 0:
    #             # 该只股票已买入,从context中清理掉
    #             del context.sell_markets[market]
    #     return
    # elif now_time >= end_sell_time:
    #     # 判断是否没有完全成交，如果没有，则强制卖出
    #     if sell_markets:
    #         for market, hold in sell_markets.items():
    #             # 执行卖出下单
    #             context.order(context.symbol(market), -hold['amount'])
    #             Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(hold['amount']))
    #         context.sell_markets = {}
    #     return
    # return
    ''' 上面是测试设置 '''
    ''' 下面是测试设置 有效阴线卖出 逻辑7'''
    # if start_sell_time <= now_time < end_sell_time:
    #     # 循环判断连续阴线，如果成立就卖出
    #     for market, hold in sell_markets.items():
    #         open_cul = context.current([market], ['open'])
    #         close_cul = context.current([market], ['close'])
    #         if close_cul / open_cul < 1 - 0.001:
    #             if param.volume_limit > 0:
    #                 bar_limit = int(
    #                     float(context.current([market], ['volume'])) * 100 * param.volume_limit) // 100 * 100
    #                 if bar_limit >= hold['amount']:
    #                     new_sell_num = hold['amount']
    #                     hold['amount'] = hold['amount'] - new_sell_num
    #                 else:
    #                     new_sell_num = bar_limit
    #                     hold['amount'] = hold['amount'] - new_sell_num
    #             else:
    #                 new_sell_num = hold['amount']
    #
    #             # 执行卖出下单
    #             context.order(context.symbol(market), -new_sell_num)
    #             Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(new_sell_num))
    #     deepcopy_sell_markets = deepcopy(sell_markets)
    #     for market, hold in deepcopy_sell_markets.items():
    #         if hold['amount'] == 0:
    #             # 该只股票已买入,从context中清理掉
    #             del context.sell_markets[market]
    #     return
    # elif now_time >= end_sell_time:
    #     # 判断是否没有完全成交，如果没有，则强制卖出
    #     if sell_markets:
    #         for market, hold in sell_markets.items():
    #             # 执行卖出下单
    #             context.order(context.symbol(market), -hold['amount'])
    #             Log.info("卖出的股票代码是：" + str(market) + "卖出的数量是：" + str(hold['amount']))
    #         context.sell_markets = {}
    #     return
    # return
    ''' 上面是测试设置 '''


def handle_data(context, data):
    handle_buy(context=context, data=data)
    # 新的卖出逻辑
    handle_sell(context=context, data=data)
