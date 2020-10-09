# # coding:gbk
# import wenfu
#
# '''
# 装饰器：
#     连接讯投API，并向文福提供交易接口
# 主要方法：
#     策略方法：（1）设置手续费函数(回测使用)
#             （2）周期执行调度函数
# '''
#
# def set_comminssion_xt(ContextInfo):
#     '''
#     设置手续费函数（回测使用）
#     :return:
#     '''
#     '''
#     从文福的设置手续费方法中获取commissionType、commissionList
#     --commissionType：number，可选值：
#       0：按比例
#       1：按每手（股）
#     --commissionList：list
#       open_tax：买入印花税
#       close_tax：卖出印花税
#       open_commission：开仓手续费;
#       close_commission：平仓（平昨）手续费
#       close_tdaycommission：平今手续费
#       min_commission：最少手续费
#     '''
#     comminssion = wenfu.set_comminssion()
#     commissionType = comminssion['commissionType']
#     commissionList = comminssion['commissionList']
#     print("设置回测手续费为：")
#     print(commissionType)
#     print(commissionList)
#     # 使用迅投 3.2.2(6) 函数进行设置
#     ContextInfo.set_commission(commissionType, commissionList)
#
#
#
#
# def schedule_function():
#     '''
#     周期执行调度函数（没有接口，只能在基本信息中设置，详情见迁移文档）
#     :return:
#     '''
#     pass
#
#
