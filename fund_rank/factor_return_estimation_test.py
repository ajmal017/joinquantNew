#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:43:08 2020

@author: yunongwu
"""

import numpy as np
import pandas as pd
from MySQLdb import connect, cursors
import scipy.optimize as sco
import configparser
import os
import socket

#%%
def construct_db_connections():
    hostname = socket.gethostname()
    hostname_prefix = hostname.split('-')[0]
    is_server = hostname_prefix in ('dev00', 'online00', 'online01')

    if not is_server:
        """
        In local environment, you should put a configuration file named '.db_config.ini' in the current directory.
        The configuration file content format must be:
            [DEFAULT]
            user=<YOUR_DB_USERNAME>
            password=<YOUR_PASSWORD>
            schema=<YOUR_OWN_SCHEMA>

        NOTE: Please do NOT commit the configuration file to svn
        """
        pwd = os.getcwd()
        db_config_file_path = os.path.join(pwd, '.db_config.ini')
        assert os.path.exists(db_config_file_path), "db config file %s is not existed" % db_config_file_path
        parser = configparser.ConfigParser()
        parser.read(db_config_file_path)
        user = parser['DEFAULT']['user']
        password = parser['DEFAULT']['password']
        zj_schema = parser['DEFAULT']['schema']
        db_host = 'dev.zhijuninvest.com'
    else:
        user = 'cronjob'
        password = 'ZyYADQjvh68UjySZ'
        db_host = '172.17.0.1'
        zj_schema = 'zj_data'

    jy_conn = connect(host=db_host, port=3306, user=user, passwd=password, db="JYDB", charset="utf8mb4", cursorclass=cursors.DictCursor)
    zj_conn = connect(host=db_host, port=3306, user=user, passwd=password, db=zj_schema, charset="utf8mb4", cursorclass=cursors.DictCursor)

    return jy_conn, zj_conn

#%%
def get_trading_days(start, end):
    
    conn = connect(host = "dev.zhijuninvest.com", user = "ywu", port = 3306,
                   passwd = "6GEL1YDeoPqPV4Fo", db = "JYDB", charset = "utf8mb4", 
                   cursorclass = cursors.DictCursor, autocommit = True)  
        
    cursor = conn.cursor()
    
    try:
        cursor.execute("""    
        SELECT TradingDate TradingDay
        FROM QT_TradingDayNew
            WHERE TradingDate >= '%s' 
            and TradingDate <= '%s'
            AND IfTradingDay = 1
            AND SecuMarket = 83
        """ % (start, end)
        )
        tradingday = cursor.fetchall()
        if not tradingday:
            return None
        tradingday = pd.DataFrame(list(tradingday), columns=['TradingDay'])
        tradingday = tradingday.sort_values(by='TradingDay')
        return tradingday
    finally:
        cursor.close()
        
#%%
def get_codes(start, end):
    conn = connect(host = "dev.zhijuninvest.com", user = "ywu", port = 3306,
                   passwd = "6GEL1YDeoPqPV4Fo", db = "JYDB", charset = "utf8mb4", 
                   cursorclass = cursors.DictCursor, autocommit = True)  
        
    cursor = conn.cursor()
    try:
        cursor.execute("""   
        SELECT distinct SecuCode
        FROM zj_data.FM_FactorExposure
        WHERE TradingDay >= '%s'
        AND TradingDay <= '%s'
        """ % (start, end)
        )
        codes = cursor.fetchall()
        if not codes:
            return None
        codes = pd.DataFrame(list(codes), columns=['SecuCode'])
        codes = sorted(codes['SecuCode'].tolist())
        return codes
    finally:
        cursor.close()
    
            
#%%
def market_universe_stock(date, factor_list):
    
    conn = connect(host = "dev.zhijuninvest.com", user = "ywu", port = 3306,
                   passwd = "6GEL1YDeoPqPV4Fo", db = "JYDB", charset = "utf8mb4", 
                   cursorclass = cursors.DictCursor, autocommit = True)  
        
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            SELECT SecuCode, TradingDay, log_ret, sqrtmarketcap, marketcap, 
            country, beta, momentum, size, earnings_yield, residual_vol, growth, 
            book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, 
            ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, ind_11,
            ind_12, ind_13, ind_14, ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, 
            ind_21, ind_22, ind_23, ind_24, ind_25, ind_26, ind_27
            FROM zj_data.FM_Cleaned_Data
            WHERE TradingDay = '%s'
            """ % (date)
            )
        result_set = cursor.fetchall()
        if not result_set:
            return None
        
        market_universe = pd.DataFrame(list(result_set), 
                columns=['SecuCode', 'TradingDay', 'log_ret', 'sqrtmarketcap', 
                         'marketcap'] + factor_list)
        market_universe[['log_ret', 'sqrtmarketcap', 'marketcap'] + factor_list] = \
        market_universe[['log_ret', 'sqrtmarketcap', 'marketcap'] + factor_list].astype('float')
        # sum_cap = np.sum(market_universe['marketcap'])
        # market_universe['weight'] = market_universe['marketcap'] / sum_cap
        # market_universe['weight'] = market_universe['weight'].astype('float')
        
        return market_universe
                        
    finally:
        
        cursor.close()
        

def get_critical_date(code_list, conn):

#    code_list = df.SecuCode.unique().tolist()
    query = """
    select SecuCode, ListedDate
    from JYDB.SecuMain
        where SecuCode in %s
        and SecuCategory = 1
        and SecuMarket in (83, 90)
    """ % (str(tuple(code_list)))

    listed_dates = pd.read_sql(query, conn)
    listed_dates['CriticalDates'] = listed_dates['ListedDate'] + pd.DateOffset(months=12)
    critical_dates = listed_dates.drop('ListedDate', axis = 1)

    return critical_dates

#%%
def implied_factor_return(df_all, df, y, x):

    df = df.sort_values(by = 'TradingDay')
    df_all = df_all.sort_values(by = 'TradingDay')

    beta = pd.DataFrame()
    constraint = pd.DataFrame()
    r2 = pd.DataFrame()
    adjr2 = pd.DataFrame()
    resid = pd.DataFrame()
    resid_all = pd.DataFrame()
    yhat = pd.DataFrame()
    yhat_all = pd.DataFrame()
    tscore = pd.DataFrame()

    for t in df.TradingDay.unique():
        print(t)
        data_t = df.loc[df.TradingDay == t, : ]
        data_t_all = df_all.loc[df_all.TradingDay == t, : ]
        r = np.matrix(data_t[y]).T
        X = np.matrix(data_t[x])
        r_all = np.matrix(data_t_all[y]).T
        X_all = np.matrix(data_t_all[x])
        stock_list = data_t['SecuCode'].unique().tolist()
        stock_num = len(stock_list)
        regression_weight = data_t['sqrtmarketcap']/(data_t['sqrtmarketcap'].sum())
        W = np.matrix(np.diag(regression_weight))
        q1 = np.repeat(0, 11)
        q2 = []

        for ind in ind_list:
            ind_w = data_t[data_t[ind] == 1]['marketcap'].sum()/(data_t['marketcap'].sum())
            q2.append(ind_w)

        q = np.concatenate((q1,np.asarray(q2)), axis = 0)
        q = np.matrix(q)
        pin = np.linalg.pinv(2 * X.T * W * X)
        f = (pin - pin * q.T * np.linalg.pinv(q * pin * q.T) * q * pin) * 2 * X.T * W * r

        params = pd.DataFrame(f.T, columns = factor_list)
        params['TradingDay'] = t

        cons = q * f
        cons = pd.DataFrame(cons)
        cons['TradingDay'] = t
        cons = cons.rename(columns = {0 : 'ind_cap_sum'})

        res = r - X * f
        resid_sq = np.multiply((r - X * f), (r - X * f))
        res = np.append(res, resid_sq, axis = 1)
        res = pd.DataFrame(res, columns = ['wls_resid', 'wls_resid2'])
        res = data_t[['TradingDay','SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
              merge(res, left_index = True, right_index = True)

        tot = r - np.matrix(regression_weight) * r
        r_sq = 1 - np.matrix(regression_weight) * resid_sq/(np.matrix(regression_weight) * np.multiply(tot, tot))
        r_sq = pd.DataFrame(r_sq, columns = ['r2'])
        r_sq['TradingDay'] = t

        r_adj = 1 - (1 - r_sq.r2) * (stock_num - 1)/(stock_num - 1 - factor_num)
        r_adj = pd.DataFrame(r_adj)
        r_adj = r_adj.rename(columns = {'r2': 'adjr2'})
        r_adj['TradingDay'] = t

        yt = pd.DataFrame(X * f, columns = ['wls_yhat'])
        yt = data_t[['TradingDay', 'SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
             merge(yt, left_index = True, right_index = True)

        yt_all = pd.DataFrame(X_all * f, columns = ['wls_yhat'])
        yt_all = data_t_all[['TradingDay', 'SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
             merge(yt_all, left_index = True, right_index = True)

        tvalue = np.divide(f.T, np.sqrt(np.divide(resid_sq.sum()/(stock_num - 38), np.power(X - X.mean(0),2).sum(0))))
        tvalue = pd.DataFrame(tvalue, columns = factor_list)
        tvalue['TradingDay'] = t
        tvalue = tvalue.drop('country', axis = 1)

        res_all = r_all - X_all * f
        resid_sq_all = np.multiply((r_all - X_all * f), (r_all - X_all * f))
        res_all = np.append(res_all, resid_sq_all, axis = 1)
        res_all = pd.DataFrame(res_all, columns = ['wls_resid', 'wls_resid2'])
        res_all = data_t_all[['TradingDay','SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
              merge(res_all, left_index = True, right_index = True)

        beta = beta.append(params)
        constraint = constraint.append(cons)
        resid = resid.append(res)
        resid_all = resid_all.append(res_all)
        r2 = r2.append(r_sq)
        adjr2 = adjr2.append(r_adj)
        yhat = yhat.append(yt)
        yhat_all = yhat_all.append(yt_all)
        tscore = tscore.append(tvalue)

    return beta, constraint, resid, resid_all, r2, adjr2, yhat, yhat_all, tscore


#%%
def implied_factor_return2(df_all, df, y, x):

    df = df.sort_values(by = 'TradingDay')
    df_all = df_all.sort_values(by = 'TradingDay')

    beta = pd.DataFrame()
    constraint = pd.DataFrame()
    r2 = pd.DataFrame()
    adjr2 = pd.DataFrame()
    resid = pd.DataFrame()
    resid_all = pd.DataFrame()
    yhat = pd.DataFrame()
    yhat_all = pd.DataFrame()
    tscore = pd.DataFrame()

    for t in df.TradingDay.unique():
        print(t)
        data_t = df.loc[df.TradingDay == t, : ]
        data_t_all = df_all.loc[df_all.TradingDay == t, : ]
        r = np.matrix(data_t[y]).T
        X = np.matrix(data_t[x])
        r_all = np.matrix(data_t_all[y]).T
        X_all = np.matrix(data_t_all[x])
        stock_list = data_t['SecuCode'].unique().tolist()
        stock_num = len(stock_list)
        regression_weight = data_t['sqrtmarketcap']/(data_t['sqrtmarketcap'].sum())
        W = np.matrix(np.diag(regression_weight))
        q1 = np.repeat(0, 11)
        q2 = []

        for ind in ind_list:
            ind_w = data_t[data_t[ind] == 1]['marketcap'].sum()/(data_t['marketcap'].sum())
            q2.append(ind_w)

        q = np.concatenate((q1,np.asarray(q2)), axis = 0)
        q = np.matrix(q)
#        pin = np.linalg.pinv(2 * X.T * W * X)
#        f = np.linalg.pinv(X.T*X+q.T*q)*X.T*r
        f = np.linalg.pinv(X.T*W*X+q.T*q)*X.T*W*r

        params = pd.DataFrame(f.T, columns = factor_list)
        params['TradingDay'] = t

        cons = q * f
        cons = pd.DataFrame(cons)
        cons['TradingDay'] = t
        cons = cons.rename(columns = {0 : 'ind_cap_sum'})

        res = r - X * f
        resid_sq = np.multiply((r - X * f), (r - X * f))
        res = np.append(res, resid_sq, axis = 1)
        res = pd.DataFrame(res, columns = ['wls_resid', 'wls_resid2'])
        res = data_t[['TradingDay','SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
              merge(res, left_index = True, right_index = True)

        tot = r - np.matrix(regression_weight) * r
        r_sq = 1 - np.matrix(regression_weight) * resid_sq/(np.matrix(regression_weight) * np.multiply(tot, tot))
        r_sq = pd.DataFrame(r_sq, columns = ['r2'])
        r_sq['TradingDay'] = t

        r_adj = 1 - (1 - r_sq.r2) * (stock_num - 1)/(stock_num - 1 - factor_num)
        r_adj = pd.DataFrame(r_adj)
        r_adj = r_adj.rename(columns = {'r2': 'adjr2'})
        r_adj['TradingDay'] = t

        yt = pd.DataFrame(X * f, columns = ['wls_yhat'])
        yt = data_t[['TradingDay', 'SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
             merge(yt, left_index = True, right_index = True)

        yt_all = pd.DataFrame(X_all * f, columns = ['wls_yhat'])
        yt_all = data_t_all[['TradingDay', 'SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
             merge(yt_all, left_index = True, right_index = True)

        tvalue = np.divide(f.T, np.sqrt(np.divide(resid_sq.sum()/(stock_num - 38), np.power(X - X.mean(0),2).sum(0))))
        tvalue = pd.DataFrame(tvalue, columns = factor_list)
        tvalue['TradingDay'] = t
        tvalue = tvalue.drop('country', axis = 1)

        res_all = r_all - X_all * f
        resid_sq_all = np.multiply((r_all - X_all * f), (r_all - X_all * f))
        res_all = np.append(res_all, resid_sq_all, axis = 1)
        res_all = pd.DataFrame(res_all, columns = ['wls_resid', 'wls_resid2'])
        res_all = data_t_all[['TradingDay','SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
              merge(res_all, left_index = True, right_index = True)

        beta = beta.append(params)
        constraint = constraint.append(cons)
        resid = resid.append(res)
        resid_all = resid_all.append(res_all)
        r2 = r2.append(r_sq)
        adjr2 = adjr2.append(r_adj)
        yhat = yhat.append(yt)
        yhat_all = yhat_all.append(yt_all)
        tscore = tscore.append(tvalue)

    return beta, constraint, resid, resid_all, r2, adjr2, yhat, yhat_all, tscore

#%%
def implied_factor_return3(df_all, df, y, x):

    df = df.sort_values(by = 'TradingDay')
    df_all = df_all.sort_values(by = 'TradingDay')

    beta = pd.DataFrame()
    constraint = pd.DataFrame()
    r2 = pd.DataFrame()
    adjr2 = pd.DataFrame()
    resid = pd.DataFrame()
    resid_all = pd.DataFrame()
    yhat = pd.DataFrame()
    yhat_all = pd.DataFrame()
#    tscore = pd.DataFrame()

    for t in df.TradingDay.unique():
        print(t)
        data_t = df.loc[df.TradingDay == t, : ]
        data_t_all = df_all.loc[df_all.TradingDay == t, : ]
        r = np.array(data_t[y])
        X = np.array(data_t[x])
        r_all = np.array(data_t_all[y])
        X_all = np.array(data_t_all[x])
        stock_list = data_t['SecuCode'].unique().tolist()
        stock_num = len(stock_list)
        regression_weight = data_t['sqrtmarketcap']/(data_t['sqrtmarketcap'].sum())
        W = np.array(regression_weight)
#        W = np.matrix(np.diag(regression_weight))
        q1 = np.repeat(0, 11)
        q2 = []

        for ind in ind_list:
            ind_w = data_t[data_t[ind] == 1]['marketcap'].sum()/(data_t['marketcap'].sum())
            q2.append(ind_w)

        q = np.concatenate((q1,np.asarray(q2)), axis = 0)
        q = np.matrix(q)
#        pin = np.linalg.pinv(2 * X.T * W * X)
#        f = np.linalg.pinv(X.T*X+q.T*q)*X.T*r
#        f = np.linalg.pinv(X.T*W*X+q.T*q)*X.T*W*r
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.multiply(x, q))})
        f = minimize_wss(X, r, W, constraints)

        params = pd.DataFrame(f).T
        params.columns = factor_list
        params['TradingDay'] = t

        cons = np.multiply(q, f).sum()
        cons = pd.DataFrame({'ind_cap_sum': [cons]})
        cons['TradingDay'] = t

        res = r - np.dot(X, f)
        resid_sq = np.multiply(r - np.dot(X, f), r - np.dot(X, f))
#        res = np.append(res, resid_sq, axis = 1)
        res = pd.DataFrame({'wls_resid': res, 'wls_resid2': resid_sq})
        res = data_t[['TradingDay','SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
              merge(res, left_index = True, right_index = True)

        tot = r - np.multiply(regression_weight, r)
        r_sq = 1 - np.dot(regression_weight, resid_sq)/(np.dot(regression_weight, np.multiply(tot, tot)))
        r_sq = pd.DataFrame({'r2': [r_sq]})
        r_sq['TradingDay'] = t

        r_adj = 1 - (1 - r_sq.r2.iloc[0]) * (stock_num - 1)/(stock_num - 1 - factor_num)
        r_adj = pd.DataFrame({'adjr2': [r_adj]})
#        r_adj = r_adj.rename(columns = {'r2': 'adjr2'})
        r_adj['TradingDay'] = t

        yt = pd.DataFrame(np.dot(X, f), columns = ['wls_yhat'])
        yt = data_t[['TradingDay', 'SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
             merge(yt, left_index = True, right_index = True)

        yt_all = pd.DataFrame(np.dot(X_all, f), columns = ['wls_yhat'])
        yt_all = data_t_all[['TradingDay', 'SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
             merge(yt_all, left_index = True, right_index = True)
        '''
        tvalue = np.divide(f.T, np.sqrt(np.divide(resid_sq.sum()/(stock_num - 38), np.power(X - X.mean(0),2).sum(0))))
        tvalue = pd.DataFrame(tvalue, columns = factor_list)
        tvalue['TradingDay'] = t
        tvalue = tvalue.drop('country', axis = 1)
        '''
        res_all = r_all - np.dot(X_all, f)
        resid_sq_all = np.multiply((r_all - np.dot(X_all, f)), (r_all - np.dot(X_all, f)))
#        res_all = np.append(res_all, resid_sq_all, axis = 1)
        res_all = pd.DataFrame({'wls_resid': res_all, 'wls_resid2': resid_sq_all})
#        res_all = pd.DataFrame(res_all, columns = ['wls_resid', 'wls_resid2'])
        res_all = data_t_all[['TradingDay','SecuCode']].reset_index().drop(columns = ['index'], axis = 1).\
              merge(res_all, left_index = True, right_index = True)

        beta = beta.append(params)
        constraint = constraint.append(cons)
        resid = resid.append(res)
        resid_all = resid_all.append(res_all)
        r2 = r2.append(r_sq)
        adjr2 = adjr2.append(r_adj)
        yhat = yhat.append(yt)
        yhat_all = yhat_all.append(yt_all)
#        tscore = tscore.append(tvalue)

    return beta, constraint, resid, resid_all, r2, adjr2, yhat, yhat_all

#%%
def wss(beta, x, y, w):
    wss = np.multiply(w, (y - np.dot(x, beta.T))**2).sum()
    return wss

def minimize_wss(x, y, w, constraints):
    num_params = x.shape[1]    
    args = (x, y, w)

    result = sco.minimize(wss, num_params * [1./num_params, ], args=args,
                          method='SLSQP', bounds=None, constraints=constraints)
    
    return result['x']


#%%
'''
zj_cursor.execute("""CREATE TABLE "FM_WLS_Beta3" (
  "id"          int NOT NULL AUTO_INCREMENT  comment 'id',
  "TradingDay" datetime NOT NULL,
  "country" decimal(10, 6) NOT NULL,
  "beta" decimal(10, 6) NOT NULL,
  "momentum" decimal(10, 6) NOT NULL,
  "size" decimal(10, 6) NOT NULL,
  "earnings_yield" decimal(10, 6) NOT NULL,
  "residual_vol" decimal(10, 6) NOT NULL,
  "growth" decimal(10, 6) NOT NULL,
  "book_to_price" decimal(10, 6) NOT NULL,
  "leverage" decimal(10, 6) NOT NULL,
  "liquidity" decimal(10, 6) NOT NULL,
  "non_linear_size" decimal(10, 6) NOT NULL,
  "ind_0" decimal(10, 6) NOT NULL,
  "ind_1" decimal(10, 6) NOT NULL,
  "ind_2" decimal(10, 6) NOT NULL,
  "ind_3" decimal(10, 6) NOT NULL,
  "ind_4" decimal(10, 6) NOT NULL,
  "ind_5" decimal(10, 6) NOT NULL,
  "ind_6" decimal(10, 6) NOT NULL,
  "ind_7" decimal(10, 6) NOT NULL,
  "ind_8" decimal(10, 6) NOT NULL,
  "ind_9" decimal(10, 6) NOT NULL,
  "ind_10" decimal(10, 6) NOT NULL,
  "ind_11" decimal(10, 6) NOT NULL,
  "ind_12" decimal(10, 6) NOT NULL,
  "ind_13" decimal(10, 6) NOT NULL,
  "ind_14" decimal(10, 6) NOT NULL,
  "ind_15" decimal(10, 6) NOT NULL,
  "ind_16" decimal(10, 6) NOT NULL,
  "ind_17" decimal(10, 6) NOT NULL,
  "ind_18" decimal(10, 6) NOT NULL,
  "ind_19" decimal(10, 6) NOT NULL,
  "ind_20" decimal(10, 6) NOT NULL,
  "ind_21" decimal(10, 6) NOT NULL,
  "ind_22" decimal(10, 6) NOT NULL,
  "ind_23" decimal(10, 6) NOT NULL,
  "ind_24" decimal(10, 6) NOT NULL,
  "ind_25" decimal(10, 6) NOT NULL,
  "ind_26" decimal(10, 6) NOT NULL,
  "ind_27" decimal(10, 6) NOT NULL,
  "update_time" timestamp NULL comment '更新时间'  DEFAULT  CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY       (ID),
  UNIQUE KEY "TradingDay" ("TradingDay")
) ENGINE=InnoDB DEFAULT CHARSET=utf8;""")
zj_conn.commit()


zj_cursor.execute("""CREATE TABLE "FM_WLS_Yhat" (
    "id"          int NOT NULL AUTO_INCREMENT  comment 'id',
    "TradingDay" datetime NOT NULL,
    "SecuCode" char(6) NOT NULL,
    "wls_yhat" decimal(10, 6) NOT NULL,
    "update_time" timestamp NULL comment '更新时间'  DEFAULT  CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY       (ID),
    UNIQUE KEY "TradingDay_SecuCode" ("TradingDay", "SecuCode")
) ENGINE=InnoDB DEFAULT CHARSET=utf8;""")
zj_conn.commit()

zj_cursor.execute("""CREATE TABLE "FM_WLS_Resid_Var" (
  "id"          int NOT NULL AUTO_INCREMENT  comment 'id',
  "TradingDay" datetime NOT NULL,
  "SecuCode" char(6) NOT NULL,
  "wls_resid" decimal(10, 6) NOT NULL,
  "wls_resid2" decimal(10, 6) NOT NULL,
  "update_time" timestamp NULL comment '更新时间'  DEFAULT  CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY       (ID),
  UNIQUE KEY "SecuCode_TradingDay" ("SecuCode", "TradingDay")
) ENGINE=InnoDB DEFAULT CHARSET=utf8;""")
zj_conn.commit()

zj_cursor.execute("""CREATE TABLE "FM_WLS_R2" (
    "id"          int NOT NULL AUTO_INCREMENT  comment 'id',
    "TradingDay" datetime NOT NULL,
    "r2" decimal(10, 6) NOT NULL,
    "update_time" timestamp NULL comment '更新时间'  DEFAULT  CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY       (ID),
    UNIQUE KEY "TradingDay" ("TradingDay")
) ENGINE=InnoDB DEFAULT CHARSET=utf8;""")
zj_conn.commit()


zj_cursor.execute("""CREATE TABLE "FM_WLS_AdjR2" (
    "id"          int NOT NULL AUTO_INCREMENT  comment 'id',
    "TradingDay" datetime NOT NULL,
    "adjr2" decimal(10, 6) NOT NULL,
    "update_time" timestamp NULL comment '更新时间'  DEFAULT  CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY       (ID),
    UNIQUE KEY "TradingDay" ("TradingDay")
) ENGINE=InnoDB DEFAULT CHARSET=utf8;""")
zj_conn.commit()
'''

#%%
if __name__ == '__main__':

    # get the datdabase connections for the furthur operations
    jy_conn, zj_conn = construct_db_connections()

    jy_conn.autocommit = True
    zj_conn.autocommit = True
    jy_cursor = jy_conn.cursor()
    zj_cursor = zj_conn.cursor()
    
    ind_list = ['ind_'+ str(x) for x in range(0,28)]
    factor_list = ['country', 'beta', 'momentum', 'size', 'earnings_yield',
                   'residual_vol', 'growth', 'book_to_price', 'leverage', 
                   'liquidity', 'non_linear_size'] + ind_list
    factor_num = len(factor_list)
    
    start = pd.to_datetime('2002-01-04')   
    end = pd.to_datetime('2020-03-24')
    tradingdays = get_trading_days(start, end)    
#    date = pd.to_datetime('2017-01-20')
    code_list = get_codes(start, end)
    critical_dates = get_critical_date(code_list, jy_conn)
    
    for date in tradingdays['TradingDay']:
        print(date)
        model_ret_for_regression = market_universe_stock(date, factor_list)
        model_ret_for_regression = model_ret_for_regression.sort_values(by = ['SecuCode', 'TradingDay'])
    
        model_ret_for_regression2 = pd.merge(model_ret_for_regression, critical_dates, on = 'SecuCode', how = 'left')
        # Only keep the data where TradingDay is larger than critical dates
        model_ret_for_regression2 = model_ret_for_regression2.loc[model_ret_for_regression2['TradingDay'] >= model_ret_for_regression2['CriticalDates'],]
        [wls_beta, wls_constraint, wls_resid, wls_resid_all, wls_r2, wls_adjr2, wls_yhat, wls_yhat_all] = implied_factor_return3(model_ret_for_regression, model_ret_for_regression2,
                                                                         'log_ret', factor_list)
        
        # factor returns
        values = list(zip(wls_beta['TradingDay'], wls_beta['country'], wls_beta['beta'],
                      wls_beta['momentum'], wls_beta['size'], wls_beta['earnings_yield'],
                      wls_beta['residual_vol'], wls_beta['growth'], wls_beta['book_to_price'],
                      wls_beta['leverage'], wls_beta['liquidity'], wls_beta['non_linear_size'], wls_beta['ind_0'],
                      wls_beta['ind_1'], wls_beta['ind_2'], wls_beta['ind_3'], wls_beta['ind_4'],
                      wls_beta['ind_5'], wls_beta['ind_6'], wls_beta['ind_7'], wls_beta['ind_8'],
                      wls_beta['ind_9'], wls_beta['ind_10'], wls_beta['ind_11'], wls_beta['ind_12'],
                      wls_beta['ind_13'], wls_beta['ind_14'], wls_beta['ind_15'], wls_beta['ind_16'],
                      wls_beta['ind_17'], wls_beta['ind_18'], wls_beta['ind_19'], wls_beta['ind_20'],
                      wls_beta['ind_21'], wls_beta['ind_22'], wls_beta['ind_23'], wls_beta['ind_24'],
                      wls_beta['ind_25'], wls_beta['ind_26'], wls_beta['ind_27']))

        zj_cursor.executemany('replace ywu.FM_WLS_Beta3 (TradingDay, country, beta, momentum, size, \
                                 earnings_yield, residual_vol, growth, book_to_price, leverage, \
                                 liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, \
                                 ind_7, ind_8, ind_9, ind_10, ind_11, ind_12, ind_13, ind_14, \
                                 ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, \
                                 ind_23, ind_24, ind_25, ind_26, ind_27) value (%s, %s, %s, %s, \
                                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                                 %s, %s, %s, %s)', values)
        zj_conn.commit()
        
        # residuals
        values = list(zip(wls_resid_all['TradingDay'], wls_resid_all['SecuCode'], wls_resid_all['wls_resid'],
                          wls_resid_all['wls_resid2']))

        zj_cursor.executemany('replace ywu.FM_WLS_Resid_Var (TradingDay, SecuCode, wls_resid, \
                                  wls_resid2) value (%s, %s, %s, %s)', values)

        zj_conn.commit()
        
        # predicted log returns
        values = list(zip(wls_yhat_all['TradingDay'], wls_yhat_all['SecuCode'], wls_yhat_all['wls_yhat']))

        zj_cursor.executemany('replace ywu.FM_WLS_Yhat (TradingDay, SecuCode, wls_yhat) \
                                  value (%s, %s, %s)', values)

        zj_conn.commit()
        
        # R2
        values = list(zip(wls_r2['TradingDay'], wls_r2['r2']))

        zj_cursor.executemany('replace ywu.FM_WLS_R2 (TradingDay, r2) \
                                  value (%s, %s)', values)

        zj_conn.commit()
        
        # Adjusted R2
        values = list(zip(wls_adjr2['TradingDay'], wls_adjr2['adjr2']))

        zj_cursor.executemany('replace ywu.FM_WLS_AdjR2 (TradingDay, adjr2) \
                                  value (%s, %s)', values)

        zj_conn.commit()
        

        