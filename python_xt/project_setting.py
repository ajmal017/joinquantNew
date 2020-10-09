#@Time    : 2020/6/23 14:10
#@Author  : XuWenFu

# @Time    : 2020/4/24 10:02 上午
# @Author  : XuWenFu


import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
# 设置pandas包显示行数
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
# 为了直观的显示数字，不采用科学计数法
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
# pandas.options.display.max_columns = None
# pandas.options.display.max_rows = None


############################################################
# 代码所需文件所在路径
############################################################

basic_path = '/mnt/aipaasdata/alg_data/data/'
# AI智能驱动模型保存路径
aiquant_model_path = basic_path + 'aiquant_data/result/aiquant_model/'
# 日志文件所在目录
log_path = basic_path + 'aiquant_data/log/'

############################################################
##文件所在目录位置
############################################################

# 最原始文件所在基础路径
org_file_path = basic_path + 'aiquant_data/org_datas/'

benchmark = org_file_path + 'benchmark.pkl'  ##沪深300的数据 benchmark.pkl

factors = org_file_path + 'factors.pkl'  ##因子数据 factors.pkl

history = org_file_path + 'history.pkl'  ##历史回测数据 history.pkl

merged = org_file_path + 'merged.pkl'  ##市场状态数据 market_state.csv

merged_factors = org_file_path + 'merged_factors.pkl'  ##合并后的数据 merged.pkl

market_state = org_file_path + 'market_state.csv'  ###合并后和因子合并的数据 merged_factors.pkl

state_proba = org_file_path + 'state_proba.pkl'  # 新的市场状态状态，博士改后的

# 清洗完毕，保存的文件所在路径和文件名
cleaned_file_path = basic_path + 'aiquant_data/cleaned_datas/'

cleaned_factors = cleaned_file_path + 'cleaned_factors.pkl'  ##因子数据 factors.pkl

cleaned_history = cleaned_file_path + 'cleaned_history.pkl'  ##历史回测数据 history.pkl

cleaned_merged_factors = cleaned_file_path + 'cleaned_merged_factors.pkl'  ##合并后的数据 merged.pkl
cleaned_instruments=cleaned_file_path+'cleaned_instruments.pkl'
# cleaned_market_state=cleaned_file_path+'market_state.csv' ###合并后和因子合并的数据 merged_factors.pkl

# 保存在每个市场状态下，有哪些交易日数据，数据格式pick，list格式,此数据会在直接在模型中用，无需再进行数据集成
cleaned_mkt_tradings_name = cleaned_file_path + 'cleaned__mkt_trading_dates_lst.pkl'

final_model_use_datas = cleaned_file_path + 'final_model_use_datas.pkl'


share_data='/mnt/aipaasdata/alg_data/share/data.pkl'##模型使用的数据

#backtest testfile
test_file=basic_path+'aiquant_data/log/test.pkl'
share_path=basic_path+'/share/2/'

score='/mnt/aipaasdata/alg_data/data/aiquant_data/score.csv'

stock_blacklist='/mnt/aipaasdata/alg_data/data/aiquant_data/setting/stock_blacklist' ###黑名单的文件

ea_basic_fit_data='/mnt/aipaasdata/alg_data/data/aiquant_data/ea_basic_fit_data/' #保存ea训练完毕的公式和表达式，未进行清洗的，初步的结果。
ea_result='/mnt/aipaasdata/alg_data/data/aiquant_data/ea_back_test_result/'