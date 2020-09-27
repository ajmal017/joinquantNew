"""
@ Author: 谭泽丹
@ Create File Date: 20200708
@ Version :2.0.0
"""
import websocket
import pandas as pd
import json
from time import sleep
import datetime



class JzStrategy:
    def __init__(self, name, ip):
        super().__init__()
        self.name = name
        self.order_count = 0
        self.orders = pd.DataFrame(columns=['code', 'price', 'vol', 'direction', 'open_close', 'strategy_name'])
        self.cmd_dict = {}
        self.empty_header = {}
        self.init()
        self.ws = None
        self.enable_trade = False
        self.enable_print = False
        self.connect(ip)
        print("服务已连接")
        print("监控页面:", ip.split(':')[0], ':1226')

    def init(self):
        self.cmd_dict['req_hold'] = 'OnHold'
        self.cmd_dict['req_fund'] = 'OnFund'
        self.cmd_dict['req_trades'] = 'OnTrades'
        self.cmd_dict['req_orders'] = 'OnOrders'
        self.cmd_dict['req_cancelable'] = 'OnCancelable'
        # self.cmd_dict['order'] = 'OnOrderInsert'
        self.cmd_dict['order'] = 'OnRtnOrder'
        self.cmd_dict['order_error'] = 'OnOrderInsert'
        self.cmd_dict['order_cancel'] = 'OnOrderCancel'
        self.cmd_dict['req_price'] = 'OnReqPrice'
        self.cmd_dict['req_order_strategy'] = 'OnOrderStrategy'
        # self.empty_header['req_trades'] = ["资金账号","营业部号","客户号","席位号","股东代码","证券代码","市场类型","证券名称","定位串","委托合同号","币种","成交状态","交易类型","价格类型","成交日期","成交时间","委托数量","委托价格","成交数量","成交价格","撤单数量","成交金额"]

    def connect(self, ip):
        self.ws = websocket.WebSocket()
        self.ws.connect('ws://' + ip)

    def show(self, message):
        if self.enable_print:
            print(message)

    def process_data(self, receive_data, target_cmd):
        if isinstance(receive_data, str):
            self.show(receive_data)
            return
        start_ind = 0
        search_rd = receive_data.decode('utf-8', 'ignore')
        while len(search_rd) > 0:
            # self.show(search_rd)
            split_index = search_rd[start_ind:].find('}{')
            if split_index != -1:
                end_ind = split_index + 1
            else:
                end_ind = len(search_rd)
            json_str_temp = search_rd[:end_ind]
            try:
                json_dict = json.loads(json_str_temp)
                result_temp = self.process_target_json(json_dict, target_cmd)
                if result_temp is not None:
                    return result_temp
            except:
                print('error')
                print(json_str_temp)
                return
            search_rd = search_rd[end_ind:]

    def process_target_json(self, json_dict, target_cmd):
        if not json_dict:
            return
        cmd = json_dict['cmd']
        if cmd != target_cmd:
            if cmd == 'OnRspError':
                result = json_dict['data']
                return result
            self.show('不匹配指令，不处理:')
            self.show(cmd)
            return
        self.show('收到服务器数据 指令类型:')
        self.show(cmd)
        if isinstance(json_dict['data'], list):
            result = pd.DataFrame(json_dict['data'])
        else:
            if json_dict['data'] == 'empty':
                result = pd.DataFrame(columns=self.empty_header['req_trades'])
            else:
                result = json_dict['data']
        return result

    def req_info(self, cmd):
        message = {'cmd': cmd}
        json_str = json.dumps(message).encode()
        self.show('传递json:')
        self.show(json_str)
        self.ws.send(json_str)
        sleep(0.2)
        if cmd in self.cmd_dict:
            receive_data = self.ws.recv()
            result = self.process_data(receive_data, self.cmd_dict[cmd])
            return result

    def clean_contract(self, df):
        if isinstance(df, pd.DataFrame) and ('合约' in df.columns):
            return df[~df.isnull()['合约']]
        else:
            return df

    def req_hold(self):
        self.hold_df = self.clean_contract(self.req_info('req_hold'))
        return self.hold_df

    def req_fund(self):
        self.fund_df = self.clean_contract(self.req_info('req_fund'))
        return self.fund_df

    def req_trades(self):
        self.trades_df = self.clean_contract(self.req_info('req_trades'))
        return self.trades_df

    def req_orders(self):
        self.orders_df = self.clean_contract(self.req_info('req_orders'))
        return self.orders_df

    def req_cancelable(self):
        orders_df = self.req_orders()
        self.cancelable_df = orders_df[orders_df['挂单状态'] == 51].reset_index(drop=True)
        return self.cancelable_df

    def req_order_strategy(self):
        self.os_df = self.req_info('req_order_strategy')
        if (len(self.os_df) == 0) or (self.os_df is None):
            self.os_df = pd.DataFrame(columns=['委托合同号', '策略名称'])
        return self.os_df

    def get_report_df(self):
        orders_df = self.req_orders()
        trades_df = self.req_trades()
        os = self.req_order_strategy()
        try:
            strategy_orders = pd.merge(orders_df, os, on=['委托合同号'], how='outer')
            strategy_trades = pd.merge(trades_df, os, on=['委托合同号'], how='outer')
        except Exception as e:
            print(e)
            return
        return strategy_orders, strategy_trades

    def send_order(self, code, price, vol, direction, open_close, strategy_name=None):
        if self.enable_trade is False:
            print('请开启交易开关以免误交易')
            return
        if strategy_name is None:
            strategy_name = self.name + '_' + str(self.order_count)
        message = {'cmd': 'order', 'code': str(code), 'price': price, 'vol': vol, 'direction': direction,
                   'open_close': open_close, 'strategy_name': strategy_name}
        print('发送报单', message)
        json_str = json.dumps(message).encode()
        self.ws.send(json_str)
        self.order_count += 1
        self.orders = self.orders.append(pd.DataFrame(
            {'code': [str(code)], 'price': price, 'vol': vol, 'direction': direction, 'open_close': open_close,
             'strategy_name': strategy_name}), ignore_index=True)
        sleep(0.5)
        receive_data = self.ws.recv()
        print(receive_data.decode('utf-8', 'ignore'))
        result = self.process_data(receive_data, self.cmd_dict['order'])
        if result is None:
            result = self.process_data(receive_data, 'OnOrderInsert')
        # return result['委托合同号'].values[0]
        return result

    def order_cancel(self, FrontID, SessionID, orderRef, code):
        if self.enable_trade is False:
            print('请开启交易开关以免误交易')
            return
        message = {'cmd': 'order_cancel', 'FrontID': FrontID, 'SessionID': SessionID, 'orderRef': orderRef,
                   'code': code}
        json_str = json.dumps(message).encode()
        self.ws.send(json_str)
        sleep(0.1)
        receive_data = self.ws.recv()
        result = self.process_data(receive_data, 'OnOrderCancel')
        return result

    def req_price(self, code):
        if isinstance(code, list):
            message = {'cmd': 'req_price', 'code': code}
        else:
            message = {'cmd': 'req_price', 'code': str(code)}
        json_str = json.dumps(message).encode()
        self.ws.send(json_str)
        sleep(0.5)
        receive_data = self.ws.recv()
        result = self.process_data(receive_data, 'CMD_PRICE_RESULT')
        return result
