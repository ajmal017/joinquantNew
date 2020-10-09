# coding:gbk

import param
from functools import partial
import datetime
from trade_function import order, get_order, get_open_orders, cancel_order, symbol
from data_function import current, can_trade, history, history_date, get_positions, get_portfolio
from handle import *
from trade import initialize_wf, before_trading_start_wf
from logger import logger
from get_data import read_y_values, get_last_trade_date, planedorders

'''
装饰器：
    连接讯投API，并向文福提供交易接口
主要方法：
    基础方法：（1）初始化函数
            （2）事件驱动-主逻辑函数
            （3）盘前处理函数
'''


def get_stocklist(context):
    # 获取订阅股票池
    #instruments = ['002687.SZ', '300344.SZ', '601766.SH', '601601.SH', '300491.SZ', '601186.SH', '600028.SH', '300169.SZ', '300690.SZ', '300242.SZ', '300374.SZ', '300593.SZ', '600463.SH', '300648.SZ', '002858.SZ', '601155.SH', '300264.SZ', '300687.SZ', '603288.SH', '600837.SH', '603667.SH', '300345.SZ', '603002.SH', '300720.SZ', '603908.SH', '000668.SZ', '002494.SZ', '000858.SZ', '300654.SZ', '601166.SH', '300165.SZ', '002931.SZ', '603696.SH', '002558.SZ', '002810.SZ', '300240.SZ', '603966.SH', '603917.SH', '600031.SH', '600731.SH', '300750.SZ', '002825.SZ', '002136.SZ', '002066.SZ', '002580.SZ', '600802.SH', '600080.SH', '600513.SH', '600066.SH', '600051.SH', '300161.SZ', '600113.SH', '300112.SZ', '300003.SZ', '000023.SZ', '000705.SZ', '600192.SH', '300405.SZ', '300542.SZ', '002210.SZ', '601009.SH', '300645.SZ', '603477.SH', '300543.SZ', '300589.SZ', '600319.SH', '300587.SZ', '603086.SH', '600883.SH', '000776.SZ', '600893.SH', '600048.SH', '300407.SZ', '002202.SZ', '000707.SZ', '300163.SZ', '002456.SZ', '600276.SH', '600539.SH', '300619.SZ', '603136.SH', '300688.SZ', '603321.SH', '300640.SZ', '603028.SH', '600398.SH', '300421.SZ', '002676.SZ', '600156.SH', '300497.SZ', '600311.SH', '002760.SZ', '002856.SZ', '601238.SH', '002509.SZ', '600152.SH', '300736.SZ', '002017.SZ', '300471.SZ', '300086.SZ', '300004.SZ', '002259.SZ', '300260.SZ', '300693.SZ', '300637.SZ', '603813.SH', '002175.SZ', '300615.SZ', '002501.SZ', '002529.SZ', '000637.SZ', '300023.SZ', '002052.SZ', '002723.SZ', '300534.SZ', '603615.SH', '002024.SZ', '600029.SH', '300551.SZ', '600359.SH', '600735.SH', '000333.SZ', '300335.SZ', '300046.SZ', '000783.SZ', '300312.SZ', '601818.SH', '300277.SZ', '300631.SZ', '600130.SH', '002524.SZ', '600660.SH', '603041.SH', '300518.SZ', '603009.SH', '002417.SZ', '603800.SH', '600599.SH', '300105.SZ', '300585.SZ', '600078.SH', '600405.SH', '000002.SZ', '300228.SZ', '600397.SH', '000576.SZ', '603895.SH', '601933.SH', '000716.SZ', '300124.SZ', '002134.SZ', '002432.SZ', '300592.SZ', '600365.SH', '002870.SZ', '002575.SZ', '300626.SZ', '601688.SH', '300556.SZ', '300126.SZ', '300278.SZ', '002729.SZ', '600556.SH', '002865.SZ', '300449.SZ', '601088.SH', '300084.SZ', '600830.SH', '600237.SH', '300703.SZ', '002072.SZ', '002896.SZ', '600727.SH', '300356.SZ', '600354.SH', '300122.SZ', '603388.SH', '600462.SH', '300072.SZ', '000419.SZ', '300431.SZ', '600778.SH', '600309.SH', '300160.SZ', '002289.SZ', '002660.SZ', '000509.SZ', '002272.SZ', '300013.SZ', '300107.SZ', '000893.SZ', '000790.SZ', '300076.SZ', '603268.SH', '600538.SH', '603822.SH', '600900.SH', '000868.SZ', '300280.SZ', '000611.SZ', '600222.SH', '002765.SZ', '300472.SZ', '600796.SH', '600030.SH', '603726.SH', '002354.SZ', '600520.SH', '002142.SZ', '603729.SH', '600436.SH', '002248.SZ', '603809.SH', '601169.SH', '002875.SZ', '002714.SZ', '603286.SH', '300030.SZ', '600016.SH', '601668.SH', '300444.SZ', '300157.SZ', '603903.SH', '300498.SZ', '600011.SH', '002196.SZ', '603006.SH', '600600.SH', '300647.SZ', '603396.SH', '002150.SZ', '603580.SH', '603289.SH', '000001.SZ', '300453.SZ', '603488.SH', '600980.SH', '601600.SH', '601988.SH', '002667.SZ', '603335.SH', '603937.SH', '601877.SH', '600188.SH', '002054.SZ', '600958.SH', '002921.SZ', '600191.SH', '000731.SZ', '002536.SZ', '603617.SH', '002021.SZ', '000856.SZ', '300089.SZ', '300555.SZ', '603535.SH', '300517.SZ', '601628.SH', '300404.SZ', '300192.SZ', '600706.SH', '300268.SZ', '300489.SZ', '000963.SZ', '002873.SZ', '600692.SH', '600115.SH', '600470.SH', '600766.SH', '600889.SH', '600838.SH', '002278.SZ', '300397.SZ', '002576.SZ', '300535.SZ', '300211.SZ', '001979.SZ', '600689.SH', '002044.SZ', '603655.SH', '000702.SZ', '600833.SH', '000017.SZ', '002213.SZ', '300707.SZ', '300162.SZ', '002606.SZ', '600532.SH', '300283.SZ', '300097.SZ', '601939.SH', '603208.SH', '603183.SH', '601989.SH', '002198.SZ', '300306.SZ', '300659.SZ', '300246.SZ', '300490.SZ', '600867.SH', '300214.SZ', '600240.SH', '002843.SZ', '000835.SZ', '600186.SH', '600361.SH', '603050.SH', '603380.SH', '600025.SH', '002849.SZ', '000955.SZ', '002809.SZ', '600000.SH', '300179.SZ', '300234.SZ', '300584.SZ', '600165.SH', '002356.SZ', '000018.SZ', '300380.SZ', '600637.SH', '300635.SZ', '300563.SZ', '600768.SH', '300093.SZ', '300462.SZ', '603022.SH', '300270.SZ', '603059.SH', '002027.SZ', '600821.SH', '002077.SZ', '601919.SH', '601998.SH', '600015.SH', '601012.SH', '600112.SH', '000721.SZ', '603320.SH', '002872.SZ', '600095.SH', '300025.SZ', '300509.SZ', '603829.SH', '002624.SZ', '600678.SH', '603506.SH', '002917.SZ', '603110.SH', '603722.SH', '002748.SZ', '002236.SZ', '300042.SZ', '300717.SZ', '600213.SH', '002629.SZ', '002836.SZ', '300295.SZ', '000691.SZ', '600898.SH', '600209.SH', '300680.SZ', '600448.SH', '300371.SZ', '000785.SZ', '000338.SZ', '002045.SZ', '300029.SZ', '002584.SZ', '600235.SH', '300636.SZ', '300818.SZ', '002103.SZ', '603658.SH', '603859.SH', '300478.SZ', '300095.SZ', '002112.SZ', '300139.SZ', '300598.SZ', '600747.SH', '000506.SZ', '600540.SH', '002427.SZ', '002943.SZ', '002862.SZ', '300632.SZ', '300090.SZ', '002778.SZ', '603703.SH', '600297.SH', '002231.SZ', '603037.SH', '300164.SZ', '600128.SH', '000014.SZ', '002919.SZ', '300402.SZ', '600741.SH', '300722.SZ', '300071.SZ', '300650.SZ', '300218.SZ', '600543.SH', '000755.SZ', '002908.SZ', '603200.SH', '600592.SH', '603767.SH', '600438.SH', '002883.SZ', '002817.SZ', '600212.SH', '002890.SZ', '000812.SZ', '603683.SH', '300148.SZ', '002755.SZ', '002347.SZ', '600809.SH', '000816.SZ', '300655.SZ', '300103.SZ', '002205.SZ', '002808.SZ', '002076.SZ', '600036.SH', '600560.SH', '603536.SH', '002591.SZ', '002758.SZ', '601225.SH', '002105.SZ', '600519.SH', '300354.SZ', '601328.SH', '600810.SH', '600018.SH', '603880.SH', '002164.SZ', '000554.SZ', '603269.SH', '603127.SH', '300210.SZ', '300056.SZ', '300362.SZ', '002622.SZ', '600892.SH', '002696.SZ', '600249.SH', '600721.SH', '300074.SZ', '002220.SZ', '600107.SH', '300290.SZ', '600119.SH', '300235.SZ', '600099.SH', '300330.SZ', '002321.SZ', '300522.SZ', '300729.SZ', '002311.SZ', '300711.SZ', '600019.SH', '603031.SH', '002357.SZ', '600671.SH', '002120.SZ', '603721.SH', '600159.SH', '000737.SZ', '603787.SH', '002422.SZ', '601211.SH', '000609.SZ', '600009.SH', '603066.SH', '601336.SH', '002089.SZ', '601006.SH', '603500.SH', '300430.SZ', '002360.SZ', '000571.SZ', '601800.SH', '603330.SH', '603389.SH', '300501.SZ', '002694.SZ', '600104.SH', '300721.SZ', '600355.SH', '300710.SZ', '300225.SZ', '603160.SH', '600085.SH', '002296.SZ', '600203.SH', '002189.SZ', '300070.SZ', '300049.SZ', '300606.SZ', '603879.SH', '603819.SH', '600122.SH', '300015.SZ', '002026.SZ', '603709.SH', '300492.SZ', '603398.SH', '002805.SZ', '600290.SH', '600506.SH', '002316.SZ', '603499.SH', '603993.SH', '002122.SZ', '300743.SZ', '300022.SZ', '600346.SH', '603178.SH', '000596.SZ', '000538.SZ', '600328.SH', '601360.SH', '002835.SZ', '002731.SZ', '300254.SZ', '603161.SH', '000595.SZ', '300032.SZ', '300475.SZ', '600371.SH', '601888.SH', '300419.SZ', '000890.SZ', '002869.SZ', '300205.SZ', '600585.SH', '300712.SZ', '603757.SH', '603970.SH', '300515.SZ', '002886.SZ', '603139.SH', '002209.SZ', '600634.SH', '002860.SZ', '300730.SZ', '300581.SZ', '002803.SZ', '603266.SH', '000159.SZ', '300505.SZ', '603778.SH', '300281.SZ', '603045.SH', '603922.SH', '300092.SZ', '002740.SZ', '601398.SH', '300526.SZ', '603088.SH', '002455.SZ', '002012.SZ', '002569.SZ', '300697.SZ', '300540.SZ', '600615.SH', '600257.SH', '600196.SH', '300069.SZ', '002715.SZ', '002415.SZ', '600887.SH', '002816.SZ', '002762.SZ', '300167.SZ', '000708.SZ', '600677.SH', '300106.SZ', '002304.SZ', '601857.SH', '603227.SH', '300716.SZ', '300239.SZ', '300445.SZ', '300275.SZ', '300554.SZ', '000651.SZ', '601607.SH', '601318.SH', '603165.SH', '603725.SH', '002493.SZ', '000909.SZ', '600690.SH', '600666.SH', '603196.SH', '000613.SZ', '000786.SZ', '002871.SZ', '600455.SH', '002352.SZ', '601288.SH', '600746.SH', '601111.SH', '300733.SZ', '300152.SZ', '300334.SZ', '603633.SH', '603042.SH', '300375.SZ', '002442.SZ', '002711.SZ', '002594.SZ', '300605.SZ', '600050.SH', '600139.SH', '600606.SH', '300668.SZ', '300700.SZ', '600281.SH', '300063.SZ', '300537.SZ', '300062.SZ', '300536.SZ', '300446.SZ', '300452.SZ', '300175.SZ', '603776.SH', '002492.SZ', '300629.SZ', '002899.SZ', '300622.SZ']

    if not param.is_backtest:
        if param.start_date[:10] > '2020-09-21':
            instruments = planedorders('2020-09-21', param.end_date[:10], param.strategy_map[context.strategy_name])
        else:
            instruments = planedorders(param.start_date[:10], param.end_date[:10], param.strategy_map[context.strategy_name])
        # instruments = planedorders(param.start_date[:10], param.end_date[:10], param.strategy_map[context.strategy_name])
        instruments = list(set(instruments))
        return instruments
    else:
        if param.start_date[:10] > '2020-09-21':
            instruments = planedorders('2020-09-21', param.end_date[:10], param.strategy_map[context.strategy_name])
        else:
            instruments = planedorders(param.start_date[:10], param.end_date[:10], param.strategy_map[context.strategy_name])
        # y_values_df = read_y_values(start_date=get_last_trade_date(param.start_date[0:10], context=context),
        #                             end_date=param.end_date[0:10], positon=param.position)
        # if param.position == 'zx':
        #     instruments = list(set(y_values_df['instrument'].values.tolist()))
        #     return instruments
        # y_values_df = y_values_df.groupby(by='date').head(3)  # 按日分组，取出每天的前三名
        # instruments = y_values_df['instrument'].values.tolist()
        instruments = list(set(instruments))
        return instruments


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
    ContextInfo.get_open_orders = partial(get_open_orders, ContextInfo)
    ContextInfo.cancel_order = partial(cancel_order, ContextInfo)
    ContextInfo.current = partial(current, ContextInfo)
    ContextInfo.can_trade = partial(can_trade, ContextInfo)
    ContextInfo.history = partial(history, ContextInfo)
    ContextInfo.history_date = partial(history_date, ContextInfo)
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

    # ContextInfo.get_trade_detail_data = get_trade_detail_data
    # ContextInfo.timetag_to_datetime = timetag_to_datetime
    # ContextInfo.passorder = passorder

    print("--------------")
    print(current_dt(ContextInfo).strftime('%Y-%m-%d'))
    if param.is_backtest:
        if ContextInfo.get_datetime().strftime('%H:%M:%S') == '09:30:00':
            before_trading_start_wf(ContextInfo)
    else:
        if ContextInfo.get_datetime().strftime('%H:%M:%S') == '09:30:00':
            before_trading_start_wf(ContextInfo)
    # else:
    #      if not param.before_tag:
    #         before_trading_start_wf(ContextInfo)
    #         param.before_tag = True

    handle_data(ContextInfo, ContextInfo)


def before_trading_start():
    '''
    盘前处理函数（写入init中）
    :return:
    '''
    pass


def current_dt(ContextInfo):
    '''
    返回k线时间
    :param ContextInfo:
    :return:
    '''
    timetag = ContextInfo.get_bar_timetag(ContextInfo.barpos)
    now = timetag_to_datetime(timetag, '%Y-%m-%d %H:%M:%S')

    # print('========', now)
    return datetime.datetime.strptime(now, '%Y-%m-%d %H:%M:%S')


def timetag_to_datetime(timetag, format):
    import time
    timetag = timetag/1000
    time_local = time.localtime(timetag)
    return time.strftime(format,time_local)

