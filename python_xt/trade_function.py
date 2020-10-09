# coding:gbk
import param
#from daily_data_func.redis_connect import redis_connect
#from .settings.param import strategy_name
#from rediscluster import RedisCluster
import datetime
import math
# import wenfu

'''
装饰器：
    连接讯投API，并向文福提供交易接口
主要方法：
    交易相关方法：（1）通过股票代码生成股票对象
               （2）生成订单
               （3）调整股票仓位至占投资组合（股票＋现金）总金额的一定百分比
               （4）取消订单
               （5）获取订单
               （6）获取未成交的订单
               （7）交易/回测引擎(找找迅投里的)
'''

def symbol(symbol_str):
    '''
    通过股票代码生成股票对象列表 list
    :return:
    '''
    list = []
    list.append(symbol_str)
    return symbol_str


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
    print(type(ContextInfo.passorder))
    ContextInfo.passorder(opType, orderType, accountid, orderCode, prType, modelprice, abs(volume), ContextInfo.strategy_name, quickTrade, ContextInfo,)
    # 将初次买入时间记录下来
    # if amount > 0:
    #     if asset in ContextInfo.start_buy_list:
    #         pass
    #     else:ss
    #         ContextInfo.start_buy_list.append(asset)
    #         r = redis_connect()
    #         key_start_buy_date = strategy_name + '-' + asset
    #         r.set(key_start_buy_date, str(datetime.date.today()), ex=6048000)  # 过期时间70天



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
    deal_list = ContextInfo.get_trade_detail_data(accountid, strAccountType, strDatatype_deal)

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




def get_open_orders(ContextInfo, sid=None):
    '''
    获取未成交的订单
    *注：委托列表和成交列表中的委托号是一样的,都是这个 m_strOrderSysID 属性值。
    :return:
    '''
    '''
    print("获取未成交的订单操作开始")
    # 0. 准备数据
    # 资金账号
    accountid = ContextInfo.accID
    # 账户类型，默认股票
    strAccountType = 'STOCK'
    # 两个列表分别保存委托的委托号，和交易的委托号
    order_sysid_list = []
    deal_sysid_list = []

    # 1. 获取交易明细中委托对象列表，并将 m_strOrderSysID 保存为委托order_id列表， 调用迅投 3.2.4.2(3) 函数
    strDatatype_order = 'ORDER'
    order_list = get_trade_detail_data(accountid, strAccountType, strDatatype_order)
    for item in order_list:
        order_sysid_list.append(item.m_strOrderSysID)
    print("委托交易的委托号列表：")
    print(order_sysid_list)

    # 2. 获取交易明细中成交对象列表，并将 m_strOrderSysID 保存为成交order_id列表， 调用迅投 3.2.4.2(3) 函数
    strDatatype_deal = 'DEAL'
    deal_list = get_trade_detail_data(accountid, strAccountType, strDatatype_deal)
    for item in deal_list:
        deal_sysid_list.append(item.m_strOrderSysID)
    print("成交交易的委托号列表：")
    print(deal_sysid_list)

    # 3. 对比委托和成交的 order_id 列表，找到委托后未成交的 order_id 们
    ret_sysid_list = [item for item in order_sysid_list if item not in deal_sysid_list]
    print("未成交的委托号列表：")
    print(ret_sysid_list)

    # 4. 按要求格式返回查询结果
    print("获取未成交的订单操作完成")
    return ret_sysid_list
    '''
    pass
    return None


def cancel_order(ContextInfo, order=None):
    '''
    取消订单(尾盘时，将所有未成交的订单取消掉)
    :param order:
    :return:
    '''
    '''
    print("取消订单操作开始")
    # 0. 准备数据
    # 资金账号
    accountid = ContextInfo.accID
    # 账户类型，默认股票
    strAccountType = 'STOCK'

    # 1. 获取未成交订单
    ret_sysid_list = get_open_orders(ContextInfo)

    # 2. 查询委托是否可以撤销，如果可以就直接进行取消委托
    for item in ret_sysid_list:
        orderId = item
        # 调用迅投 3.2.4.2 (6) 函数进行查询
        if can_cancel_order(orderId, accountid, strAccountType):
            # 执行取消委托， 调用迅投 3.2.4.2 (7) 函数
            cancel(orderId, accountid, strAccountType, ContextInfo)
        else:
            print("委托号：" + orderId + "无法取消")

    print("取消订单操作完成")
    '''
    pass