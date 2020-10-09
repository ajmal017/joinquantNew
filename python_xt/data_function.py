# coding:gbk
#import 文福的模块
import param
import datetime, time
#from daily_data_func.redis_connect import redis_connect
#from settings.param import strategy_name

'''
装饰器：
    连接讯投API，并向文福提供交易接口
主要方法：
    数据操作方法：（1）获取当天股票数据
               （2）获取一定时间窗口的历史数据
               （3）该股票是否可以交易
'''

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
    df = ContextInfo.get_market_data(fields, stock_code, skip_paused=True, period=period, dividend_type=dividend_type, count=2)
    df = df[fields[0]]
    df = df[0]
    return df





def history(ContextInfo, assets, fields, bar_count, frequency):
    '''
    获取一定时间窗口的历史数据
    :return:
    '''
    # 今天的日期str
    today = ContextInfo.get_datetime().strftime('%Y-%m-%d')
    # 构建datetime对象
    open_time = datetime.datetime.strptime((today + ' 09:30:00'), '%Y-%m-%d %H:%M:%S')
    morning_time = datetime.datetime.strptime((today + ' 11:30:00'), '%Y-%m-%d %H:%M:%S')
    afternoon_time = datetime.datetime.strptime((today + ' 13:00:00'), '%Y-%m-%d %H:%M:%S')
    now_time = datetime.datetime.strptime((ContextInfo.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')

    # 计算出start_time
    if (now_time >= open_time) and (now_time <= morning_time):
        start_time = now_time - datetime.timedelta(minutes=1) * int(bar_count)
    elif now_time >= afternoon_time:
        start_time = now_time - datetime.timedelta(minutes=1) * (int(bar_count) + 90)

    start_time = start_time.strftime('%Y%m%d%H%M%S')
    # 计算出end_time
    end_time = now_time.strftime('%Y%m%d%H%M%S')

    dividend_type = 'none'

    # 使用迅投 3.2.3(17) 函数进行设置
    df = ContextInfo.get_market_data(fields, assets, start_time, end_time,
                                skip_paused=True, period=frequency, dividend_type=dividend_type, count=-1)
    return trans(df)



def history_date(ContextInfo, assets, fields, frequency):
    '''
    获取一定时间窗口的历史数据(按时间开始、结束获取)
    :return:
    '''

    # 今天的日期str
    today = ContextInfo.get_datetime().strftime('%Y-%m-%d')
    # 构建datetime对象
    open_time = datetime.datetime.strptime((today + ' 09:30:00'), '%Y-%m-%d %H:%M:%S')
    now_time = datetime.datetime.strptime((ContextInfo.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')

    # 计算出str
    start_time = open_time.strftime('%Y%m%d%H%M%S')
    end_time = now_time.strftime('%Y%m%d%H%M%S')

    dividend_type = 'none'

    # 使用迅投 3.2.3(17) 函数进行设置
    df = ContextInfo.get_market_data(fields, assets, start_time, end_time,
                                skip_paused=True, period=frequency, dividend_type=dividend_type, count=-1)
    return trans(df)


def can_trade(ContextInfo, stockcode):
    '''
    该股票是否可以交易，返回值类型是布尔型
    :param stockcode: String 股票代码
    :return: number
    '''

    # 使用迅投 3.2.3(1) 函数进行设置
    number = ContextInfo.get_last_volume(stockcode)
    if number <= 0 :
        return False
    else:
        return True

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
    #r = redis_connect()

    # 1. 查询持仓信息, 调用迅投 3.2.4.2(3) 函数
    strDatatype = 'POSITION'
    position_info_list = ContextInfo.get_trade_detail_data(accountid, strAccountType, strDatatype, ContextInfo.strategy_name)

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
        #start_buy_date = r.get(strategy_name + '-' + item.m_strInstrumentID)   # 去redis里面读取
        # start_buy_date = ''
        # position_target_dict['start_buy_date'] = datetime.datetime.strptime(start_buy_date, "%Y-%m-%d")
        position_dict[item.m_strInstrumentID] = position_target_dict

    # 3. 按要求格式返回查询结果
    print("查询持仓信息完成")
    return position_dict



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
        account_info_list = ContextInfo.get_trade_detail_data(accountid, strAccountType, strDatatype)


        print('get position',  type(ContextInfo.get_trade_detail_data))
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
        account_total_info_dict['returns'] = "%.2f" % (item.m_dPositionProfit/param.capital_base)
    print('portfolio_value:%s' % account_total_info_dict['portfolio_value'])
    print('positions_value:%s' % account_total_info_dict['positions_value'])
    print('cash:%s' % account_total_info_dict['cash'])
    # 3. 按要求格式返回查询结果
    print("查询账户总信息完成")
    return account_total_info_dict



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
    the_panel=the_panel.transpose((2,1,0))
    b = the_panel.to_frame()
    d = b.reset_index()
    d.rename(columns={'major': 'date', 'minor': 'instrument'}, inplace=True)
    return d