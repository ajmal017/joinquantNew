#@Time    : 2020/6/23 14:09
#@Author  : XuWenFu

# @Time    : 2020/6/9 3:08 下午
# @Author  : XuWenFu

###从中信获取数据
import sys
import os
import warnings
import pandas as pd
import numpy as np
import multiprocessing
# from bigdatasource.impl.bigdatasource import DataSource
# from hmmlearn.hmm import GaussianHMM
import time
import datetime
import pickle as pk
from multiprocessing import Process, Pool, Lock

warnings.filterwarnings('ignore')

factors = [1, 2, 3, 4, 5, 6, 7]  # 传入特征列表


def read_data(start_date, end_date, fields=None, table='AShareEODPrices', instruments=None):
    '''
    获取数据进行封装
    :param start_date:
    :param end_date:
    :param fields:
    :param table:
    :param instruments:
    :return:
    '''
    df = None
    if instruments and fields:
        df = DataSource(table).read(instruments, start_date=start_date, end_date=end_date, fields=fields)
    if instruments and not fields:
        df = DataSource(table).read(instruments, start_date=start_date, end_date=end_date)
    if fields and not instruments:
        df = DataSource(table).read(start_date=start_date, end_date=end_date, fields=fields)
    if not fields and not instruments:
        df = DataSource(table).read(start_date=start_date, end_date=end_date)
    if isinstance(df, pd.DataFrame) and ('date' in df.columns):
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['date', 'instrument'])
        return df
    elif isinstance(df, pd.DataFrame) and ('date' not in df.columns):
        return df
    else:
        raise IOError('未读取到相应的数据，请检查数据输入:' + str(start_date) + ' ' + str(end_date) + ' ' + str(table) + ' ' + str(
            instruments))

'''
def read_hmm(instruments=None, start_date='2010-01-04', check_file='/tmp/checkhmm.pkl'):

    def get_hmm():
        now = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        df = read_data(start_date=start_date, end_date=now, fields=['s_dq_close'],
                            table='AIndexEODPrices', instruments=instruments)
        df.rename(columns={'s_dq_close': 'close'}, inplace=True)
        df = df[['date', 'instrument', 'close']]
        df.sort_values(by=['instrument', 'date'], inplace=True)
        df = df.groupby(by=['instrument'], as_index=False).apply(lambda x: x.fillna(method='pad'))
        df = df.dropna()
        df['close'] = df.groupby(by=['instrument'])['close'].shift(1)
        df = df.dropna()
        df.sort_values(by=['date', 'instrument'], inplace=True)
        df1 = df.loc[df['instrument'] == instruments[0]]
        df1['close1'] = df.loc[df['instrument'] == instruments[1]]['close'].values
        df1['instrument1'] = df.loc[df['instrument'] == instruments[1]]['instrument'].values
        df1[['close', 'close1']] = df1[['close', 'close1']].apply(lambda x: np.log(x))
        df1.reset_index(drop=True, inplace=True)
        samples = df1[['close', 'close1']].values
        hmm_4 = GaussianHMM(n_components=6, covariance_type='full', n_iter=500000).fit(samples)
        pre = hmm_4.predict_proba(samples)
        hmmdf = pd.DataFrame(pre, columns=['factor9999', 'factor9998', 'factor9997', 'factor9996', 'factor9995',
                                           'factor9994'])
        hmmdf[hmmdf < 0.1] = 0
        hmmdf['date'] = df1['date']
        return hmmdf

    if instruments:
        instruments = instruments
    else:
        instruments = ['000931.CSI', '000932.SH']

    if not os.path.exists(check_file):
        df = get_hmm()
        df[['date', 'factor9999', 'factor9998', 'factor9997', 'factor9996', 'factor9995', 'factor9994']].to_pickle(
            check_file)
        return df
    else:
        check_df = pd.read_pickle(check_file).head(500)
        df = get_hmm()
        data_df = df['date']
        df.drop(columns=['date'], inplace=True)
        columns = [1, 2, 3, 4, 5, 6]
        df.columns = columns
        min = 1000000000
        info = ['factor9999', 'factor9998', 'factor9997', 'factor9996', 'factor9995', 'factor9994']
        for i in range(0, 6):
            for t in info:
                nor = np.linalg.norm(check_df[t] - df[i + 1].head(500))
                if nor < min:
                    min = nor
                    columns[i] = t
            info.remove(columns[i])
            min = 1000000000
    df.columns = columns
    df['date'] = data_df
    return df
'''

def read_instruments(start_date, end_date):
    '''
    读取所有的的股票代码列表
    :return:
    '''
    df = DataSource('AShareDescription').read(start_date=start_date, end_date=end_date,
                                              fields=['s_info_name', 'instrument', 's_info_delistdate'])
    df = df[df['s_info_delistdate'].isna() | df['s_info_delistdate'].isnull()]
    # df = df[~df['s_info_name'].str.contains("ST")]
    df = df[~df['instrument'].str.contains('A')]
    if isinstance(df, pd.DataFrame):
        instruments = list(set(df['instrument'].values.tolist()))
        # print('读取的股票列表有:', df.shape, len(instruments))
        return instruments
    else:
        raise IOError("未读取到相应的股票代码列表，请检查func read_instruments() or DataSource(id='AShareCodeTable').read()")


def read_benchmark_df(start_date, end_date, instruments=None):
    '''
    读取基准数据
    :return:
    '''
    instrument = ["000300.SH"] if (instruments is None) else instruments
    target_fields = ['date', 'instrument', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                     'pctchange']
    origin_fields = ['date', 'instrument', 's_dq_open', 's_dq_high', 's_dq_low', 's_dq_close',
                     's_dq_preclose', 's_dq_volume', 's_dq_amount', 's_dq_pctchange']

    benchmark = read_data(instruments=instrument, start_date=start_date, end_date=end_date,
                          table='AIndexEODPrices', fields=origin_fields)
    benchmark.rename_axis({x: y for x, y in zip(origin_fields, target_fields)}, axis=1, inplace=True)
    benchmark.reset_index(drop=True, inplace=True)
    print('benchmark的缺失数据为：', benchmark.shape, benchmark.isna().sum())
    return benchmark


def read_ohlc_df(start_date, end_date,instruments,traget_fields=None):
    '''
    读取开盘价，收盘价格，最高，最低的等数据数据
    :param instruments: 输入股票列表
    :param traget_fields: 数据读取的股票colunms  必须是字典类型
    :return: df
    '''
    if not traget_fields:
        target_fields = ['date', 'instrument', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                         'pctchange']
        origin_fields = ['date', 'instrument', 's_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 's_dq_adjclose',
                         's_dq_adjpreclose', 's_dq_volume', 's_dq_amount', 's_dq_pctchange']

    else:
        origin_fields = traget_fields.keys()
        target_fields = traget_fields.values()

    ohlc_df = read_data(instruments=instruments, start_date=start_date, end_date=end_date,
                             fields=origin_fields)
    ohlc_df.rename_axis({x: y for x, y in zip(origin_fields, target_fields)}, axis=1, inplace=True)
    ohlc_df = ohlc_df[ohlc_df['pctchange'] < 11]
    ohlc_df = ohlc_df[ohlc_df['pctchange'] > -11]
    ohlc_df['pctchange'] = (ohlc_df['pctchange']).fillna(0)
    ohlc_df.reset_index(drop=True, inplace=True)
    print('开盘收盘的缺失数据为：', ohlc_df.shape, ohlc_df.isna().sum())
    return ohlc_df


def merge_factors(start_date,end_date,instruments,factors_lts):
    merge_df = None
    print('进入合并函数内')
    data=[]
    for i in range(0, len(factors_lts)):
        df = read_data(instruments=instruments, table=str(factors_lts[i]), start_date=start_date,
                            end_date=end_date)
        data.append(df)
    for i in range(0,len(data)):
        if i ==0:
            merge_df=data[i]
        else:
            merge_df=pd.merge(merge_df,data[i],on=['date','instrument'],how='inner')
    return merge_df


def read_cleaned_factors(start_date,end_date,factors: list, instruments: list, multiprocess=True):
    '''
    读取特征列表
    :param factors: factor数字类型，【1，3，4，4，6，7..】,必须从1开始
    :return:
    '''
    if factors[0]==0:
        raise IOError('fators必须从1开始，不是从0开始，请更改')

    def mycallback(item):
        '''
        回调方法：将ea可用结果放入useful_ea_result中
        '''
        # 如果返回不是None，则开始收集数据
        if item is not None:
            # 加锁
            lock.acquire()
            retLst.append(item)
            lock.release()

    # 创建进程锁
    lock = Lock()
    # 设置进程池
    pool = Pool(30)
    # 整合成factor_table
    factors_lts = ["features_CN_STOCK_A_factor{}".format(factor) for factor in factors]
    # 判断使用多进程，还是单进程
    retLst = []
    jobs = (len(factors) - 10) // 3
    if multiprocess and jobs > 2:
        # 启用多进程读取数据
        # 计算开启多少个分组，多少个进程
        print('启动多进程')
        cpu_count = multiprocessing.cpu_count()
        jobs = cpu_count if jobs > cpu_count else jobs
        print(jobs)
        # 切分数据
        split_lts = [factors_lts[i:i + jobs] for i in range(0, jobs, len(factors) // jobs)]
        for i in range(len(split_lts)):
            pool.apply_async(func=merge_factors, args=(split_lts[i],), callback=mycallback)
        pool.close()
        pool.join()
        final_df = retLst[0]
        for i in range(1, len(retLst)):
            final_df = pd.merge(final_df, retLst[i], on=['date', 'instrument'], how='outer')
    else:  # 单进程
        final_df = merge_factors(start_date=start_date,end_date=end_date,factors_lts=factors_lts,instruments=instruments)
    final_df.reset_index(drop=True, inplace=True)
    print('factors缺失数据为：', final_df.shape, final_df.isna().sum())

    return final_df


def read_all(start_date,end_date,read_benchmark_check=True, read_instruments_check=True, read_ohlc_check=True, factors=None, cleaning_data=True,
             normalization=True):
    '''
    读取所有的数据，并且对数据进行预处理。
    :param read_instruments: 读取股票列表数据
    :param read_ohlc_df: 读取开盘价收盘价数据
    :param read_cleaned_factors: 读取特征因子数据
    :param cleaning_data: 进行数据清理。
    :param normalization: 进行特征归一化
    :return: 字典{}
    '''
    instruments=None
    factors_df=None
    data_read_ohlc_df=None
    data_benchmark_df=None
    now = datetime.datetime.now()
    if read_benchmark_check:
        data_benchmark_df = read_benchmark_df(start_date=start_date,end_date=end_date)
        print('读取benchmark_df 所用时间', datetime.datetime.now() - now)
    if read_instruments_check:
        instruments = read_instruments(start_date=start_date,end_date=end_date)
        print('读取instruments 所用时间', datetime.datetime.now() - now)
    if read_ohlc_check:
        data_read_ohlc_df =read_ohlc_df(start_date=start_date,end_date=end_date,instruments=instruments)
        print('读取read_ohlc_df 所用时间', datetime.datetime.now() - now)
    if factors:
        factors_df = read_cleaned_factors(start_date=start_date,end_date=end_date,factors=factors, instruments=instruments)
        print('读取factors_df 所用时间', datetime.datetime.now() - now)
    if cleaning_data:
        factors_df = cleaning_facotors(factors_df)
        print('清洗factors_df 所用时间', datetime.datetime.now() - now)
    if normalization:
        factors_df = normalization_df(factors_df)
        print('正则化factors_df 所用时间', datetime.datetime.now() - now)

    final_model_use_df = pd.merge(factors_df, data_read_ohlc_df, on=['date', 'instrument'], how='inner')
    final_model_use_df = pd.merge(final_model_use_df, read_hmm(), on=['date', 'instrument'], how='inner')

    print("*****************************************************")
    print('-------------------------------------')
    print('benchmark 详细信息：')
    print(data_benchmark_df.shape)
    print(data_benchmark_df.info)
    print(data_benchmark_df.isna().sum())
    print(data_benchmark_df.head(3))
    print(data_benchmark_df.tail(3))
    print('-------------------------------------')
    print('instruments 详细信息：')
    print(instruments)
    print('-------------------------------------')
    print('read_ohlc_df 详细信息：')
    print(data_read_ohlc_df.shape)
    print(data_read_ohlc_df.info)
    print(data_read_ohlc_df.isna().sum())
    print(data_read_ohlc_df.head(3))
    print(data_read_ohlc_df.tail(3))
    print('-------------------------------------')
    print('factors_df 详细信息：')
    print(factors_df.shape)
    print(factors_df.info)
    print(factors_df.isna().sum())
    print(factors_df.head(3))
    print(factors_df.tail(3))
    print('-------------------------------------')
    print('final_model_use_df 详细信息：')
    print(final_model_use_df.shape)
    print(final_model_use_df.info)
    print(final_model_use_df.isna().sum())
    print(final_model_use_df.head(3))
    print(final_model_use_df.tail(3))
    print("*****************************************************")
    result = {'benchmark': data_benchmark_df, 'instruments': instruments, 'ohlc_df': data_read_ohlc_df,
              'factors_df': factors_df, 'final_model_use_df': final_model_use_df}
    return result


def cleaning_facotors(df):
    '''
    对merged文件进行清洗，并且归一化
    :param df:
    :return: 返回清洗完毕，并且标准化的数据
    '''
    # 获取缺失数据超过20%的列，并删除这些数据
    df = df.loc[df.isna().sum(axis=1) < df.shape[1] * 0.3, df.isna().sum() < df.shape[0] * 0.04]
    df.sort_values(by=['instrument', 'date'], inplace=True)
    df = df.groupby(by=['instrument'], as_index=False).apply(lambda x: x.fillna(method='pad'))
    df = df.fillna(0)
    df.sort_values(by=['date', 'instrument'], inplace=True)
    return df


def normalization_df(df, by_date=True):
    '''
    对数据进行归一化操作
    :param df:
    :param bydate: 按照日期分组，对每日的数据进行归一化，如果是否，就对整个dataframe进行归一化
    :return: df
    '''
    df.sort_values(by=['date', 'instrument'], inplace=True)

    def cal(x):
        other = x[['date', 'instrument']]
        caldf = x.drop(columns=['date', 'instrument'])
        redf2 = (caldf - caldf.mean()) / (caldf.std())
        redf2['date'] = other['date']
        redf2['instrument'] = other['instrument']
        return redf2

    if by_date:
        df = df.groupby(by=['date'], as_index=False).apply(lambda x: cal(x))
    else:
        df = cal(df)

    return df


if __name__ == "__main__":
    '''主程序执行入口，从此处执行文件'''
    start_date='2020-06-06'
    end_date='2020-06-24'
    ins=read_instruments(start_date=start_date,end_date=end_date)
    history=read_ohlc_df(start_date=start_date,end_date=end_date,instruments=ins)
    history=history.sort_values(by=['instrument','date'])
    history['rank']=history.groupby(by='instrument')['close'].rank().astype(int)
    today=history.loc[history['date']==end_date]
    print(today)
    useful=today.loc[today['rank']<3,['instrument','rank']]
    print(useful)
    # data=pd.read_pickle('/mnt/aipaasdata/alg_data/share/data.pkl')
    # final=pd.merge(history,data,on=['date','instrument'],how='inner')
    # final.to_pickle('/mnt/aipaasdata/alg_data/share/data3.pkl')
    # print('文件保存完毕')
    # factors = [i for i in range(1, 901)]
    # ###验证hmm算法是否存在bug
    # # hmm=read_data.read_hmm()
    # # print(hmm)'/mnt/aipaasdata/alg_data/share/data.pkl'
    # # time.sleep(20)
    #
    # ####查找出相关性低的因子来
    # instruments =read_instruments(start_date=start_date,end_date=end_date)
    # read_benchmark(start_date=start_date,end_date=end_date)
    # factors_df = read_cleaned_factors(start_date=start_date,end_date=end_date,factors=factors, instruments=instruments)
    # corr_df = factors_df.corr()
    # factorslts = []
    # for columns in corr_df.columns.values.tolist():
    #     factorinfo = corr_df.loc[corr_df[columns] < 0.3].index.values.tolist()
    #     for fa in factorinfo:
    #         factorslts.append(fa)
    # final_factorlist = list(set(factorslts))
    # print('相关性低的因子有：')
    # print(final_factorlist)
    # print('相关性低的因子有：')
    # # insd = './factorlist.pkl'
    # # with open(insd, 'wb') as f:
    # #     f.truncate()
    # #     pk.dump(insd, f)
    #
    # # result = {'benchmark': benchmark_df, 'instruments': instruments, 'ohlc_df': read_ohlc_df,
    # #           'factors_df': factors_df, 'final_model_use_df': final_model_use_df}
    # ###读取所有的数据，并保存起来
    # data =read_all(start_date=start_date,end_date=end_date,factors=final_factorlist)
    # for key, values in data.items():
    #     if key == 'benchmark':
    #         values.to_pickle('./benchmark_df.pkl')
    #     if key == 'instruments':
    #         ins = './instruments.pkl'
    #         with open(ins, 'wb') as f:
    #             f.truncate()
    #             pk.dump(ins, f)
    #     if key == 'ohlc_df':
    #         values.to_pickle('./read_ohlc_df.pkl')
    #     if key == 'factors_df':
    #         values.to_pickle('./factors_df.pkl')
    #     if key == 'final_model_use_df':
    #         values.to_pickle('./final_model_use_df.pkl')

    # data = read_data.read_cleaned_factors(factors=range(1,101),instruments=['601318.SH','000858.SZ','600519.SH'])
    # read_data.read_all(factors=[1, 2,3,4,5,6,7,8,9,10,11,12,13])
    # read_data.read_instruments()
    # read_data.read_benchmark()
