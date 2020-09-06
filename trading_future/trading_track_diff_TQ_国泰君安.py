#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 16:37
# @Author  : zf
# @Site    :
# @Software: PyCharm
import datetime
import pandas as pd
import numpy as np
import time
# from jqdatasdk import *
import tkinter
import tkinter.messagebox
from tqsdk import TqApi, TqSim, TqAccount
# from configDB import *
# JOINQUANT_USER = '15168322665'
# JOINQUANT_PW = 'Juzheng2018'
# auth(JOINQUANT_USER, JOINQUANT_PW)
from email_fuction import send_email


class Trading:
    def __init__(self, api):
        self.api = api

    def get_time_allowed(self, time_allowed_lst):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[-8:]
        if now > '00:00:00' and now < '03:00:00':
            now = str(int(now[1]) + 24) + now[2:]
        allowed = False
        for (s_time, e_time) in time_allowed_lst:
            if now > s_time and now < e_time:
                allowed = True
                break
        return allowed

    # def get_date(self, calen, today):
    #     next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    #     if datetime.datetime.now().hour >= 15:
    #         calen.append(next_tradeday)
    #     EndDate = calen[-1]
    #     StartDate = calen[0]
    #     hq_last_date = calen[-2]
    #     return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]

    def insert_order_bk_limit(self, code, volume):
        quote = self.api.get_quote(code)
        limit_price = quote['upper_limit']
        order = None
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            order = self.api.insert_order(code, direction='BUY', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_sk_limit(self, code, volume):
        quote = self.api.get_quote(code)
        limit_price = quote['lower_limit']
        order = None
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            order = self.api.insert_order(code, direction='SELL', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_bp_limit(self, code, volume=None):
        position_account = self.api.get_position(code)
        position_short = position_account.pos_short
        order = None
        quote = self.api.get_quote(code)
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if position_short:
            if volume:
                position_short = min(position_short, volume)
            quote = self.api.get_quote(code)
            limit_price = quote['upper_limit']
            if limit_price > 0:
                order = self.api.insert_order(code, direction='BUY', offset='CLOSE', volume=position_short,
                                         limit_price=limit_price)
                a = 0
                while (order.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_bpk_limit(self, code, volume):
        position_account = self.api.get_position(code)
        position_short = position_account.pos_short
        order = None
        quote = self.api.get_quote(code)
        limit_price = quote['upper_limit']
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            if position_short:
                order_bp = self.api.insert_order(code, direction='BUY', offset='CLOSE', volume=position_short, limit_price=limit_price)
                a = 0
                while (order_bp.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order_bp.status, order_bp.volume_left))
            order = self.api.insert_order(code, direction='BUY', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_spk_limit(self, code, volume):
        position_account = self.api.get_position(code)
        position_long = position_account.pos_long
        order = None
        quote = self.api.get_quote(code)
        limit_price = quote['lower_limit']
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            if position_long:
                order_sp = self.api.insert_order(code, direction='SELL', offset='CLOSE', volume=position_long, limit_price=limit_price)
                a = 0
                while (order_sp.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order_sp.status, order_sp.volume_left))
            order = self.api.insert_order(code, direction='SELL', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_sp_limit(self, code, volume=None):
        position_account = self.api.get_position(code)
        position_long = position_account.pos_long
        order = None
        quote = self.api.get_quote(code)
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if position_long:
            if volume:
                position_long = min(position_long, volume)
            quote = self.api.get_quote(code)
            limit_price = quote['lower_limit']
            if limit_price > 0:
                order = self.api.insert_order(code, direction='SELL', offset='CLOSE', volume=position_long,
                                         limit_price=limit_price)
                a = 0
                while (order.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order


def get_alert_info(info_txt):
    tkinter.messagebox.showinfo('提示', info_txt)


if __name__ == '__main__':
    trd_stime = '2020-07-14 09:00:30'
    trd_etime = '2020-07-15 02:30:00'
    api = TqApi(TqAccount("G国泰君安", "85030120", "jz04282020"), web_gui=True)
    Trd = Trading(api)
    # order = Trd.insert_order_sk_limit('INE.sc2012', 1)
    # order = Trd.insert_order_bk_limit('INE.sc2007', 1)
    hold_code_lst = ['sc2008', 'sc2012']
    start_day = datetime.date.today().strftime('%Y-%m-%d')
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    receiver = ['xiahutao@163.com', '3467518502@qq.com', '897174480@qq.com']
    long_code_lst = ['sc2008']
    short_code_lst = ['sc2012']
    # long_cost_lst = [311.16]
    long_cost_lst = [271.7]
    short_cost_lst = [321]
    long_volume = [1]
    short_volume = [-1]

    lst = []
    times1 = 0
    trad = True
    while datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < trd_etime:
        api.wait_update()
        lst = []
        long_value_now = 0
        short_value_now = 0
        quote_08 = api.get_quote("INE.sc2008")
        quote_12 = api.get_quote("INE.sc2012")
        # print(quote_08)
        price_08 = quote_08.last_price
        price_12 = quote_12.last_price
        diff = price_12 - price_08

        df = pd.DataFrame([[price_08, price_12, diff]], columns=['price_08', 'price_12', 'diff'])
        print(datetime.datetime.now())
        print(df)

        if (diff < 33) and times1 < 3 and trad == True and \
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') > trd_stime:
            try:
                send_email(df, '完成1次', receiver)
                # times1 += 1
                # logger_async.log(__name__, logger_async.critical, '完成第一次交易')
            except Exception as e:
                print(str(e))
            # try:
            #     order1 = Trd.insert_order_bp_limit('INE.sc2012', 1)
            #     order2 = Trd.insert_order_sp_limit('INE.sc2008', 1)
            #     times1 += 1
            # except Exception as e:
            #     print(str(e))
            get_alert_info('价差小于33')
        # time.sleep(5)
