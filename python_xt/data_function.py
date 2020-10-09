# coding:gbk
#import �ĸ���ģ��
import param
import datetime, time
#from daily_data_func.redis_connect import redis_connect
#from settings.param import strategy_name

'''
װ������
    ����ѶͶAPI�������ĸ��ṩ���׽ӿ�
��Ҫ������
    ���ݲ�����������1����ȡ�����Ʊ����
               ��2����ȡһ��ʱ�䴰�ڵ���ʷ����
               ��3���ù�Ʊ�Ƿ���Խ���
'''

def current(ContextInfo, assets, fields):
    '''
    ��ȡ�����Ʊ����(�ӿ��̵���ǰʱ�̵Ķ��Ĺ�Ʊ�б�ĸ�������)
    ��ע������ǰʱ�̣�����end_time������
    :return:
    '''
    '''
    fields���ֶ��б�
      'open'����  'high'����  'low'����  'close'����
      'volume'���ɽ���  'amount'���ɽ���  'settle'�������
    stock_code: list ��Ʊ�����б�
    start_time��Ĭ�ϲ�������ʼʱ�䣬��ʽ '20171209' �� '20171209010101'
    end_time��Ĭ�ϲ���������ʱ�䣬��ʽ '20171209' �� '20171209010101'
    skip_paused��Ĭ�ϲ�������ѡֵ��
      true�������ͣ�ƹɣ����Զ����δͣ��ǰ�ļ۸���Ϊͣ���յļ۸�
      False��ͣ������Ϊnan
    period��Ĭ�ϲ������������ͣ�
      'tick'���ֱ���  '1d'������  '1m'��1������  '3m'��3������  '5m'��5������
      '15m'��15������  '30m'��30������  '1h'��Сʱ��  '1w'������  '1mon'������
      '1q'������  '1hy'��������  '1y'������
    dividend_type��Ĭ�ϲ�����ȱʡֵΪ 'none'������Ȩ����ѡֵ��
      'none'������Ȩ  'front'����ǰ��Ȩ  'back'�����Ȩ  'front_ratio'���ȱ���ǰ��Ȩ  'back_ratio'���ȱ����Ȩ
    count�� = -1 ����start_time �� end_time��Ч
    '''
    # ��ȡ��Ʊ���еĹ�Ʊ��ʹ��ѸͶ 3.2.2(7) ����
    stock_code = assets

    # ��ȡ����
    period = ContextInfo.period

    dividend_type = 'none'
    # ʹ��ѸͶ 3.2.3(17) ������������
    df = ContextInfo.get_market_data(fields, stock_code, skip_paused=True, period=period, dividend_type=dividend_type, count=2)
    df = df[fields[0]]
    df = df[0]
    return df





def history(ContextInfo, assets, fields, bar_count, frequency):
    '''
    ��ȡһ��ʱ�䴰�ڵ���ʷ����
    :return:
    '''
    # ���������str
    today = ContextInfo.get_datetime().strftime('%Y-%m-%d')
    # ����datetime����
    open_time = datetime.datetime.strptime((today + ' 09:30:00'), '%Y-%m-%d %H:%M:%S')
    morning_time = datetime.datetime.strptime((today + ' 11:30:00'), '%Y-%m-%d %H:%M:%S')
    afternoon_time = datetime.datetime.strptime((today + ' 13:00:00'), '%Y-%m-%d %H:%M:%S')
    now_time = datetime.datetime.strptime((ContextInfo.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')

    # �����start_time
    if (now_time >= open_time) and (now_time <= morning_time):
        start_time = now_time - datetime.timedelta(minutes=1) * int(bar_count)
    elif now_time >= afternoon_time:
        start_time = now_time - datetime.timedelta(minutes=1) * (int(bar_count) + 90)

    start_time = start_time.strftime('%Y%m%d%H%M%S')
    # �����end_time
    end_time = now_time.strftime('%Y%m%d%H%M%S')

    dividend_type = 'none'

    # ʹ��ѸͶ 3.2.3(17) ������������
    df = ContextInfo.get_market_data(fields, assets, start_time, end_time,
                                skip_paused=True, period=frequency, dividend_type=dividend_type, count=-1)
    return trans(df)



def history_date(ContextInfo, assets, fields, frequency):
    '''
    ��ȡһ��ʱ�䴰�ڵ���ʷ����(��ʱ�俪ʼ��������ȡ)
    :return:
    '''

    # ���������str
    today = ContextInfo.get_datetime().strftime('%Y-%m-%d')
    # ����datetime����
    open_time = datetime.datetime.strptime((today + ' 09:30:00'), '%Y-%m-%d %H:%M:%S')
    now_time = datetime.datetime.strptime((ContextInfo.get_datetime().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')

    # �����str
    start_time = open_time.strftime('%Y%m%d%H%M%S')
    end_time = now_time.strftime('%Y%m%d%H%M%S')

    dividend_type = 'none'

    # ʹ��ѸͶ 3.2.3(17) ������������
    df = ContextInfo.get_market_data(fields, assets, start_time, end_time,
                                skip_paused=True, period=frequency, dividend_type=dividend_type, count=-1)
    return trans(df)


def can_trade(ContextInfo, stockcode):
    '''
    �ù�Ʊ�Ƿ���Խ��ף�����ֵ�����ǲ�����
    :param stockcode: String ��Ʊ����
    :return: number
    '''

    # ʹ��ѸͶ 3.2.3(1) ������������
    number = ContextInfo.get_last_volume(stockcode)
    if number <= 0 :
        return False
    else:
        return True

def get_positions(ContextInfo=None):
    '''
    ��ѯ�ֲ���Ϣ
    :return:
    '''

    print("��ѯ�ֲ���Ϣ��ʼ")
    # 0. ׼������
    # �ʽ��˺�
    accountid = param.accountid
    # �˻����ͣ�Ĭ�Ϲ�Ʊ
    strAccountType = 'STOCK'
    #r = redis_connect()

    # 1. ��ѯ�ֲ���Ϣ, ����ѸͶ 3.2.4.2(3) ����
    strDatatype = 'POSITION'
    position_info_list = ContextInfo.get_trade_detail_data(accountid, strAccountType, strDatatype, ContextInfo.strategy_name)

    # 2. ���췵�ؽ��
    position_dict = {}
    for item in position_info_list:
        position_target_dict = {}
        position_target_dict['amount'] = item.m_nVolume
        position_target_dict['cost_basis'] = item.m_dOpenPrice
        position_target_dict['last_sale_price'] = item.m_dSettlementPrice
        position_target_dict['sid'] = item.m_strInstrumentID
        position_target_dict['last_sale_date'] = datetime.date.today()
        position_target_dict['asset'] = None
        #start_buy_date = r.get(strategy_name + '-' + item.m_strInstrumentID)   # ȥredis�����ȡ
        # start_buy_date = ''
        # position_target_dict['start_buy_date'] = datetime.datetime.strptime(start_buy_date, "%Y-%m-%d")
        position_dict[item.m_strInstrumentID] = position_target_dict

    # 3. ��Ҫ���ʽ���ز�ѯ���
    print("��ѯ�ֲ���Ϣ���")
    return position_dict



def get_portfolio(ContextInfo=None):
    '''
    ��ȡ�˻�����Ϣ
    :param self:
    :return:
    '''

    print("��ѯ�˻�����Ϣ��ʼ")
    # 0. ׼������
    # �ʽ��˺�
    accountid = param.accountid
    # �˻����ͣ�Ĭ�Ϲ�Ʊ
    strAccountType = 'STOCK'

    # 1. ��ѯ�ʽ��˻���Ϣ, ����ѸͶ 3.2.4.2(3) ����
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
    # 2. ���췵�ؽ��
    account_total_info_dict = {}
    for item in account_info_list:
        # �˻��ֽ� float
        account_total_info_dict['cash'] = item.m_dAvailable
        # �����˻�����ͳ�ƿ�ʼʱ�� datetime
        account_total_info_dict['start_date'] = datetime.datetime.strptime(param.start_date, "%Y-%m-%d %H:%M:%S")
        # �����˻�����ͳ�ƽ���ʱ�� datetime
        account_total_info_dict['end_date'] = datetime.datetime.strptime(param.end_date, "%Y-%m-%d %H:%M:%S")
        # �����˻���ʼ��� float
        account_total_info_dict['starting_cash'] = param.capital_base
        # �˻��ܼ�ֵ�������ֲ���ֵ+�ֽ� float
        account_total_info_dict['portfolio_value'] = item.m_dBalance
        # �ֲ���ֵ float
        account_total_info_dict['positions_value'] = item.m_dStockValue
        # �ֲ� dictionary
        account_total_info_dict['positions'] = ContextInfo.get_positions()
        # �ֲַ��ձ�¶ float
        account_total_info_dict['positions_exposure'] = ''
        # �˻����������ĵľ��ʲ�(������)��Ϊ��ʱ������ float
        account_total_info_dict['capital_used'] = item.m_dCommission
        # �ֲ����� float
        account_total_info_dict['pnl'] = item.m_dPositionProfit
        # �˻��ۼ����棬����10%���ص���0.1  float
        account_total_info_dict['returns'] = "%.2f" % (item.m_dPositionProfit/param.capital_base)
    print('portfolio_value:%s' % account_total_info_dict['portfolio_value'])
    print('positions_value:%s' % account_total_info_dict['positions_value'])
    print('cash:%s' % account_total_info_dict['cash'])
    # 3. ��Ҫ���ʽ���ز�ѯ���
    print("��ѯ�˻�����Ϣ���")
    return account_total_info_dict



def trans(qmt_df):
    '''
    ��ѸͶ�������������ݸ�ʽ����ת����3D->3D��
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