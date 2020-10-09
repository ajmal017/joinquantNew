# coding:gbk
import param
#from daily_data_func.redis_connect import redis_connect
#from .settings.param import strategy_name
#from rediscluster import RedisCluster
import datetime
import math
# import wenfu

'''
װ������
    ����ѶͶAPI�������ĸ��ṩ���׽ӿ�
��Ҫ������
    ������ط�������1��ͨ����Ʊ�������ɹ�Ʊ����
               ��2�����ɶ���
               ��3��������Ʊ��λ��ռͶ����ϣ���Ʊ���ֽ��ܽ���һ���ٷֱ�
               ��4��ȡ������
               ��5����ȡ����
               ��6����ȡδ�ɽ��Ķ���
               ��7������/�ز�����(����ѸͶ���)
'''

def symbol(symbol_str):
    '''
    ͨ����Ʊ�������ɹ�Ʊ�����б� list
    :return:
    '''
    list = []
    list.append(symbol_str)
    return symbol_str


def order(ContextInfo, asset, amount, position_effect=None, limit_price=None, stop_price=None, style=None):
    '''
    ���ɶ��������µ�����
    :return:
    '''

    # opType, orderType, orderCode, prType, volume, ContextInfo
    '''
      ʵ����passorder(23,1202, 'testS', '000001.SZ', 5, -1, 50000, ContextInfo)����˼���Ƕ��˺��� testS ���
      �����˺Ŷ������¼ۿ������� 50000 Ԫ��ֵ�� 000001.SZ ƽ������
    --opType��number����ѡֵ��
      23����Ʊ���룬�򻦸�ͨ�����ͨ��Ʊ����
      24����Ʊ�������򻦸�ͨ�����ͨ��Ʊ����
    --orderType���µ���ʽ����ѡֵ��
      1101�����ɡ����˺š���ͨ����/�ַ�ʽ�µ�
      1113�����ɡ����˺š����ʲ������� [0 ~ 1] ��ʽ�µ�
      1123�����ɡ����˺š����á�����[0 ~ 1]��ʽ�µ�
    --accountID���ʽ��˺ţ��µ����˺�ID���ɶ������Ŀǰ����Ĭ���� �����ʽ��˺�: '26769922'
    --orderCode���µ����루���ڹ�Ʊ���ǹ�Ʊ���룩
    --prType���µ�ѡ�����ͣ���ѡֵ��
      0����5�� 1����4�� 2����3�� 3����2�� 4����1�� 5�����¼�
      6����1�� 7����2�� 8����3�� 9����4�� 10����5��
    --modelprice��ģ���µ��۸���Ϊ����ʹ�õ�prType����ģ�����ͣ����Ը��ֶο���������д��Ĭ��Ϊ -1
    --volume���µ��������ɡ��� / Ԫ / %�� �ڻ����׵�λ���֣���Ʊ���׵�λ�ǹ�
      ���� orderType ֵ���һλȷ�� volume �ĵ�λ��
      �����µ�ʱ��1���� / ��  2����Ԫ�� 3��������%��
    --quickTrade��int���趨�Ƿ����������µ�����ѡֵ��
      0����
      1���� --����Ŀǰ�� 1
    *ע��passorder�Ƕ����һ�� K ����ȫ��������ɵ�ģ���ź�����һ�� K �ߵĵ�һ��tick������ʱ�����µ����ף�
      ����quickTrade��������Ϊ1ʱ��ֻҪ����ģ���е��õ�passorder���׺����ʹ����µ����ס�
    '''

    print("���ɶ�����")
    print("��Ʊ����Ϊ��" + asset + " ;�µ�����Ϊ��" + str(amount))

    # ����amount�������������������
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
    # ʹ��ѸͶ 3.2.4.2(1) ������order���µ�
    print(orderCode)
    print(type(ContextInfo.passorder))
    ContextInfo.passorder(opType, orderType, accountid, orderCode, prType, modelprice, abs(volume), ContextInfo.strategy_name, quickTrade, ContextInfo,)
    # ����������ʱ���¼����
    # if amount > 0:
    #     if asset in ContextInfo.start_buy_list:
    #         pass
    #     else:ss
    #         ContextInfo.start_buy_list.append(asset)
    #         r = redis_connect()
    #         key_start_buy_date = strategy_name + '-' + asset
    #         r.set(key_start_buy_date, str(datetime.date.today()), ex=6048000)  # ����ʱ��70��



def get_order(ContextInfo, order=None):
    '''
    ��ȡ�����ѳɽ��Ķ���
    :param order:
    :return:
    '''

    print("��ȡ�ѳɽ��Ķ���������ʼ")
    # 0. ׼������
    # �ʽ��˺�
    accountid = ContextInfo.accID
    print(accountid)
    # �˻����ͣ�Ĭ�Ϲ�Ʊ
    strAccountType = 'STOCK'

    # 1. ��ȡ������ϸ�гɽ������б����� m_strOrderSysID ����Ϊ�ɽ�order_id�б� ����ѸͶ 3.2.4.2(3) ����
    strDatatype_deal = 'DEAL'
    deal_list = ContextInfo.get_trade_detail_data(accountid, strAccountType, strDatatype_deal)

    # 2. ���췵�ؽ��
    deal_dict = {}
    for item in deal_list:
        deal_target_dict = {}
        deal_target_dict['date'] = item.m_strTradeDate + item.m_strTradeTime  # �ɽ�ʱ��
        deal_target_dict['amount'] = item.m_nVolume  # �ɽ���
        deal_target_dict['price'] = item.m_dPrice  # �ɽ�����
        deal_target_dict['comssion'] = item.m_dComssion  # ������
        deal_target_dict['total'] = item.m_dPrice * item.m_nVolume + item.m_dComssion  # �ܻ��ѣ� �ɽ����� * �ɽ��� + ������
        deal_dict[item.m_strInstrumentID] = deal_target_dict

    # 3. ��Ҫ���ʽ���ز�ѯ���
    print("��ȡ�ѳɽ��Ķ����������")
    return deal_dict




def get_open_orders(ContextInfo, sid=None):
    '''
    ��ȡδ�ɽ��Ķ���
    *ע��ί���б�ͳɽ��б��е�ί�к���һ����,������� m_strOrderSysID ����ֵ��
    :return:
    '''
    '''
    print("��ȡδ�ɽ��Ķ���������ʼ")
    # 0. ׼������
    # �ʽ��˺�
    accountid = ContextInfo.accID
    # �˻����ͣ�Ĭ�Ϲ�Ʊ
    strAccountType = 'STOCK'
    # �����б�ֱ𱣴�ί�е�ί�кţ��ͽ��׵�ί�к�
    order_sysid_list = []
    deal_sysid_list = []

    # 1. ��ȡ������ϸ��ί�ж����б����� m_strOrderSysID ����Ϊί��order_id�б� ����ѸͶ 3.2.4.2(3) ����
    strDatatype_order = 'ORDER'
    order_list = get_trade_detail_data(accountid, strAccountType, strDatatype_order)
    for item in order_list:
        order_sysid_list.append(item.m_strOrderSysID)
    print("ί�н��׵�ί�к��б�")
    print(order_sysid_list)

    # 2. ��ȡ������ϸ�гɽ������б����� m_strOrderSysID ����Ϊ�ɽ�order_id�б� ����ѸͶ 3.2.4.2(3) ����
    strDatatype_deal = 'DEAL'
    deal_list = get_trade_detail_data(accountid, strAccountType, strDatatype_deal)
    for item in deal_list:
        deal_sysid_list.append(item.m_strOrderSysID)
    print("�ɽ����׵�ί�к��б�")
    print(deal_sysid_list)

    # 3. �Ա�ί�кͳɽ��� order_id �б��ҵ�ί�к�δ�ɽ��� order_id ��
    ret_sysid_list = [item for item in order_sysid_list if item not in deal_sysid_list]
    print("δ�ɽ���ί�к��б�")
    print(ret_sysid_list)

    # 4. ��Ҫ���ʽ���ز�ѯ���
    print("��ȡδ�ɽ��Ķ����������")
    return ret_sysid_list
    '''
    pass
    return None


def cancel_order(ContextInfo, order=None):
    '''
    ȡ������(β��ʱ��������δ�ɽ��Ķ���ȡ����)
    :param order:
    :return:
    '''
    '''
    print("ȡ������������ʼ")
    # 0. ׼������
    # �ʽ��˺�
    accountid = ContextInfo.accID
    # �˻����ͣ�Ĭ�Ϲ�Ʊ
    strAccountType = 'STOCK'

    # 1. ��ȡδ�ɽ�����
    ret_sysid_list = get_open_orders(ContextInfo)

    # 2. ��ѯί���Ƿ���Գ�����������Ծ�ֱ�ӽ���ȡ��ί��
    for item in ret_sysid_list:
        orderId = item
        # ����ѸͶ 3.2.4.2 (6) �������в�ѯ
        if can_cancel_order(orderId, accountid, strAccountType):
            # ִ��ȡ��ί�У� ����ѸͶ 3.2.4.2 (7) ����
            cancel(orderId, accountid, strAccountType, ContextInfo)
        else:
            print("ί�кţ�" + orderId + "�޷�ȡ��")

    print("ȡ�������������")
    '''
    pass