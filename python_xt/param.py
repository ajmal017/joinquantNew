import time
# 基础资金账号:
accountid = '26769922'

# 融资融券资金账号:
marginid = '99000269'

# True为回测, False为实盘交易
is_backtest = False
trade_start_time = [time.strftime('%Y-%m-%d', time.localtime(time.time())) + ' 09:30:00', '2018-06-01 09:00:00']

# 回测起始时间 结束时间
# start_date = '2018-06-01 09:00:00'
# end_date = '2020-09-21 15:00:00'
start_date = time.strftime('%Y-%m-%d', time.localtime(time.time())) + ' 09:00:00'
end_date = time.strftime('%Y-%m-%d', time.localtime(time.time())) + ' 15:00:00'

# 回测初始资金
capital_base = 500000

# commissionLists
buy_cost = 0.0002
sell_cost = 0.0012
open_commission = 0
close_commission = 0
close_tdaycommission = 0
min_cost = 5
commissionList = [buy_cost, sell_cost, open_commission, close_commission, close_tdaycommission, min_cost]

# redis配置
host = 'redis-cluster.middleware'
port = '6379'
password = 'kuanke@jinhua..'


# # 服务器基础路径
# basic_path = '/mnt/aipaasdata/alg_data/data/'
# # 日志文件所在目录
# log_path = basic_path + 'aiquant_data/log/'
log_path = './'


# y值文件的存储路径
# ypred_file_path = 'D:\\data\\predictions_score_K100.h5'
# ypred_file_path = 'c:\\e\\data\\qe\\predictions_score_K100.h5'
ypred_file_path = 'c:\\e\\data\\z.csv'


# 回测模型所需的初始参数
backtest_args_option = {
        'position': 'citic',   # 从中信获取数据
        'backtest':True,
        'history_df': '',
        'y_prediction': '',
        'benchmark_df':'',
        'buy_size': 1,         # 买入数量，实际上是买n-1个。若不满足风控，可能一个都不买。
        'time_window': 90,     # 时间窗口长度，判断个股处在高位还是低位
        'all_max_ratio': 0.98,   # 最大持仓量
        'first_trading_date_sign': 1,    # 首日开仓标记
        'first_trading_max_hold_ratio': 0.5,    # 首日交易最大资金比率
        'hold_ratio': 2,    # 持仓压缩比，数值越大，仓位越接近
        'one_makert_max_hold_ratio': 0.32,    # 个股最大持仓比率
        'sell_markets': {},    # 初始化卖出的信息
        'buy_markets': {},     # 初始化买入的信息
        'log_set_level': 'info',    # 日志打印级别
        'log_file_path': '/tmp/',   # 日志路径
        'hold_days': 2,
        'volume_limit': 0.025,
    }
volume_limit=0.025
tick_limit = 0.5
# position='xuntou'
position='zx'

strategy_map = {
    '淘利阿尔法1号': 'ptf8b5f6accedf11e9b8c20a580a81060a',
    '智能罗伯特管家': 'ptb99659f8cedf11e98c710a580a81060a',
    'AI智能驱动': 'pt608bd9c8cedf11e985790a580a81060a',
    '时代先锋': 'pt83dc374ccedf11e981d70a580a81060a',
}
strategy_num = 4
before_tag = False