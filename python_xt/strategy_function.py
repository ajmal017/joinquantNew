# # coding:gbk
# import wenfu
#
# '''
# װ������
#     ����ѶͶAPI�������ĸ��ṩ���׽ӿ�
# ��Ҫ������
#     ���Է�������1�����������Ѻ���(�ز�ʹ��)
#             ��2������ִ�е��Ⱥ���
# '''
#
# def set_comminssion_xt(ContextInfo):
#     '''
#     ���������Ѻ������ز�ʹ�ã�
#     :return:
#     '''
#     '''
#     ���ĸ������������ѷ����л�ȡcommissionType��commissionList
#     --commissionType��number����ѡֵ��
#       0��������
#       1����ÿ�֣��ɣ�
#     --commissionList��list
#       open_tax������ӡ��˰
#       close_tax������ӡ��˰
#       open_commission������������;
#       close_commission��ƽ�֣�ƽ��������
#       close_tdaycommission��ƽ��������
#       min_commission������������
#     '''
#     comminssion = wenfu.set_comminssion()
#     commissionType = comminssion['commissionType']
#     commissionList = comminssion['commissionList']
#     print("���ûز�������Ϊ��")
#     print(commissionType)
#     print(commissionList)
#     # ʹ��ѸͶ 3.2.2(6) ������������
#     ContextInfo.set_commission(commissionType, commissionList)
#
#
#
#
# def schedule_function():
#     '''
#     ����ִ�е��Ⱥ�����û�нӿڣ�ֻ���ڻ�����Ϣ�����ã������Ǩ���ĵ���
#     :return:
#     '''
#     pass
#
#
