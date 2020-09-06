# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''

from __future__ import division
import os
import sys
print(sys.path)
sys.path.append('C:\\Users\\Administrator\\PycharmProjects\\joinquant')  # 新加入的
# sys.path.append('../')
# sys.path.append('C:\\Users\\51951\\PycharmProjects\\resRepo')  # 新加入的
from configDB import *

import time
import matplotlib.pyplot as plt
from matplotlib import style
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import copy
import numpy as np
import pandas as pd
import talib
import tkinter
import tkinter.messagebox
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from analysis.report.graphs import Graphs
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph,NextPageTemplate,PageBreak,PageBegin
from reportlab.lib.pagesizes import letter
# from data_engine.instrument.future import Future
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph,Spacer,Image,Table
from reportlab.lib.units import cm
import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfgen.canvas import Canvas

# from common.os_func import check_fold
pdfmetrics.registerFont(TTFont("SimSun", "G:/trading/SimSun.ttf"))

# auth('18610039264', 'zg19491001')
style.use('ggplot')
auth(JOINQUANT_USER, JOINQUANT_PW)


# 获取价格
def stock_price(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    q = query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == sec).order_by(
        finance.GLOBAL_IDX_DAILY.day.desc())
    temp = finance.run_query(q)[
        ['day', 'name', 'code', 'open', 'high', 'low', 'close', 'volume']] \
        .assign(day=lambda df: df.day.apply(lambda x: str(x)[:10])) \
        .rename(columns={'day': 'trade_date', 'code': 'stock_cpde'})
    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)].sort_values(['trade_date'])
    return temp


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    if sec[:1] == 'P':
        print(sec)
    if sec in ['CU8888.XSGE', 'SN8888.XSGE', 'PB8888.XSGE', 'NI8888.XSGE', 'AL8888.XSGE', 'AU8888.XSGE', 'ZN8888.XSGE',
               'SC8888.XINE', 'AG8888.XSGE']:
        if sec == 'CU8888.XSGE':
            symble = 'HG'
        elif sec == 'ZN8888.XSGE':
            symble = 'ZSD'
        elif sec == 'SN8888.XSGE':
            symble = 'SND'
        elif sec == 'PB8888.XSGE':
            symble = 'PBD'
        elif sec == 'NI8888.XSGE':
            symble = 'NID'
        elif sec == 'AL8888.XSGE':
            symble = 'AHD'
        elif sec == 'SC8888.XINE':
            symble = 'CL'
        elif sec == 'AG8888.XSGE':
            symble = 'SI'
        else:
            symble = 'GC'
        temp = finance.run_query(query(finance.FUT_GLOBAL_DAILY).filter(
            finance.FUT_GLOBAL_DAILY.code == symble, finance.FUT_GLOBAL_DAILY.day >= sday,
            finance.FUT_GLOBAL_DAILY.day <= eday))
        temp = temp\
            .rename(columns={'day': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))[
            ['open', 'high', 'low', 'close', 'date_time', 'volume']]

    else:
        temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='pre',
                         count=None).reset_index() \
            .rename(columns={'index': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))

    temp['stock_code'] = sec
    return temp


def trans_heng_float(x):
    if x == '--':
        x = None
    return x


def KDJ(data, N=9, M1=3, M2=3):
    datelen = len(data)
    data = data[['date_time', 'open', 'high', 'low', 'close']]
    array = np.array(data)
    kdjarr = []
    k_lst = []
    d_lst = []
    j_lst = []

    for i in range(datelen):
        if i - N < 0:
            b = 0
        else:
            b = i - N + 1
        rsvarr = array[b:i + 1, 0:5]
        try:
            rsv = (float(rsvarr[-1, -1]) - float(min(rsvarr[:, 3]))) / (
                    float(max(rsvarr[:, 2])) - float(min(rsvarr[:, 3]))) * 100
            if i == 0:
                k = rsv
                d = rsv
            else:
                k = 1 / float(M1) * rsv + (float(M1) - 1) / M1 * float(kdjarr[-1][2])
                d = 1 / float(M2) * k + (float(M2) - 1) / M2 * float(kdjarr[-1][3])
            j = 3 * k - 2 * d
        except Exception as e:
            k = 50
            d = 50
            j = 50
            rsv = 50
        k_lst.append(k)
        d_lst.append(d)
        j_lst.append(j)
        kdjarr.append(list((rsvarr[-1, 0], rsv, k, d, j)))

    return k_lst, d_lst, j_lst


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    code_lst = temp.symbol.tolist()
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic, code_lst


class PDFGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.file_path = 'G:/trading/'
        self.title_style = ParagraphStyle(name="TitleStyle", fontSize=48, alignment=TA_LEFT,)
        self.sub_title_style = ParagraphStyle(name="SubTitleStyle", fontSize=32,
                                              textColor=colors.HexColor(0x666666), alignment=TA_LEFT, )
        self.content_style = ParagraphStyle(name="ContentStyle", fontSize=18, leading=25, spaceAfter=20,
                                            underlineWidth=1, alignment=TA_LEFT, )
        self.foot_style = ParagraphStyle(name="FootStyle", fontSize=14, textColor=colors.HexColor(0xB4B4B4),
                                         leading=25, spaceAfter=20, alignment=TA_CENTER, )
        self.table_title_style = ParagraphStyle(name="TableTitleStyle", fontSize=20, leading=25,
                                                spaceAfter=10, alignment=TA_LEFT, )
        self.sub_table_style = ParagraphStyle(name="SubTableTitleStyle", fontSize=16, leading=25,
                                                spaceAfter=10, alignment=TA_LEFT, )
        self.basic_style = TableStyle([('FONTNAME', (0, 0), (-1, -1), 'ping'),
                                       ('FONTSIZE', (0, 0), (-1, -1), 12),
                                       ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                       ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                       # 'SPAN' (列,行)坐标
                                       ('SPAN', (1, 0), (3, 0)),
                                       ('SPAN', (1, 1), (3, 1)),
                                       ('SPAN', (1, 2), (3, 2)),
                                       ('SPAN', (1, 5), (3, 5)),
                                       ('SPAN', (1, 6), (3, 6)),
                                       ('SPAN', (1, 7), (3, 7)),
                                       ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                                       ])
        self.common_style = TableStyle([('FONTNAME', (0, 0), (-1, -1), 'ping'),
                                      ('FONTSIZE', (0, 0), (-1, -1), 12),
                                      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                      ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                      ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                                     ])

    def genTaskPDF(self, df_hold, df_other):
        styles = getSampleStyleSheet()
        normalStyle = copy.deepcopy(styles['Normal'])
        normalStyle.fontName = 'SimSun'
        story = []
        story.append(
            Graphs.draw_title('kd_背离_%s' % datetime.datetime.now().strftime('%Y%m%d')))
        story.append(Spacer(0, 0.5 * cm))

        story.append(Paragraph('在持仓KD背离: ', normalStyle))
        story.append(Spacer(0, 0.2 * cm))
        data = [tuple(df_hold.columns)] + [tuple(x.to_dict().values()) for idx, x in df_hold.iterrows()]
        story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                         col_width=[80] + [70] * (len(df_hold.columns) - 1)))
        story.append(Spacer(0, 0.5 * cm))

        story.append(Paragraph('非在持仓KD背离: ', normalStyle))
        story.append(Spacer(0, 0.2 * cm))
        data = [tuple(df_other.columns)] + [tuple(x.to_dict().values()) for idx, x in df_other.iterrows()]
        story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                       col_width=[80] + [70] * (len(df_other.columns) - 1)))
        story.append(Spacer(0, 0.5 * cm))


        doc = SimpleDocTemplate(self.file_path + self.filename + ".pdf", pagesize=letter)
        doc.build(story)


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour > 18:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    listDATE = calen[-125]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, listDATE, StartDate, str(hq_last_date)[:10]


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    t0 = time.time()
    fold_data = 'G:/trading/'
    # data = get_all_securities(types=['futures'])

    start_day = '2017-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    calen = get_trade_days(count=3)
    calen = list(calen)
    if datetime.datetime.now().hour < 9:
        end_day = calen[-2].strftime('%Y-%m-%d')
    # end_day = '2020-04-20'
    mod = 0
    hold_code_lst = ['RU', 'P', 'AU', 'C', 'CU', 'SC', 'Y', 'CF', 'I', 'M', 'OI', 'ZN', 'AG', 'IC', 'IH', 'IF', 'JD', 'SA', 'NI']
    normalize_code_future, index_code_lst = get_normal_future_index_code()
    method = 1
    n = 1  # 回测周期
    period = '1d'
    method_lst = [('week', 'week'), ('day', 'day')]
    k1_lst = [(20, 30)]  # kd下限
    k2_lst = [(70, 80)]  # kd上限
    road_period = 4  # 进N日最高最低价周期
    kd_road_period = road_period-1
    if mod == 0:
        k1 = (15, 35)
        k2 = (65, 85)
    else:
        k1 = (20, 30)
        k2 = (70, 80)
    df_lst = []
    hq_dict = {}
    vol_dict = {}
    for index_code in index_code_lst:
        symble = normalize_code_future[index_code]
        data_daily = stock_price_cgo(symble, start_day, end_day)[
            ['date_time', 'open', 'high', 'low', 'close', 'stock_code', 'volume']]
        data_daily['vol_average'] = data_daily['volume'].shift(1).rolling(window=30).mean()
        if len(data_daily) < 30:
            continue
        if index_code == 'P':
            A = 0
        hq_dict[index_code] = data_daily
        vol_dict[index_code] = [data_daily.vol_average.tolist()[-1]]
    vol_df = pd.DataFrame(vol_dict)
    vol_df = vol_df.T
    vol_df.columns = ['volume_ave']
    vol_df = vol_df[vol_df['volume_ave'] >= 50000]
    # print(vol_df)
    index_code_lst = vol_df.index.values
    index_code_lst = list(index_code_lst)
    index_code_lst.extend(hold_code_lst)
    index_code_lst = set(index_code_lst)

    for index_code in index_code_lst:
        print(index_code)
        data_daily = hq_dict[index_code][
            ['date_time', 'open', 'high', 'low', 'close', 'stock_code']]
        data_daily['time'] = pd.to_datetime(data_daily['date_time'])
        data_daily.index = data_daily['time']
        data_daily = data_daily.drop(['time'], axis=1)
        data_week = data_daily.resample('W').last()
        data_week['date_time'] = data_daily['date_time'].resample('W').last()
        data_week['open'] = data_daily['open'].resample('W').first()
        data_week['high'] = data_daily['high'].resample('W').max()
        data_week['low'] = data_daily['low'].resample('W').min()
        data_week['close'] = data_daily['close'].resample('W').last()
        data_week = data_week.dropna()
        data_week['k_week'], data_week['d_week'], data_week['j_week'] = KDJ(data_week, 9, 3, 3)

        data_daily['k_day'], data_daily['d_day'], data_daily['j_day'] = KDJ(data_daily, 9, 3, 3)
        data_daily['h_k_day'] = talib.MAX(data_daily['k_day'].shift(1).values, kd_road_period)
        data_daily['l_k_day'] = talib.MIN(data_daily['k_day'].shift(1).values, kd_road_period)
        data_daily['h_d_day'] = talib.MAX(data_daily['d_day'].shift(1).values, kd_road_period)
        data_daily['l_d_day'] = talib.MIN(data_daily['d_day'].shift(1).values, kd_road_period)
        data_daily['h_day'] = talib.MAX(data_daily['high'].values, road_period)
        data_daily['l_day'] = talib.MIN(data_daily['low'].values, road_period)
        data_week['h_k_week'] = talib.MAX(data_week['k_week'].shift(1).values, kd_road_period)
        data_week['l_k_week'] = talib.MIN(data_week['k_week'].shift(1).values, kd_road_period)
        data_week['h_d_week'] = talib.MAX(data_week['d_week'].shift(1).values, kd_road_period)
        data_week['l_d_week'] = talib.MIN(data_week['d_week'].shift(1).values, kd_road_period)
        data_week['h_week'] = talib.MAX(data_week['high'].values, road_period)
        data_week['l_week'] = talib.MIN(data_week['low'].values, road_period)

        # data_daily.to_csv(fold_data + symble + '_data_daily.csv')
        data_daily['day_kd_b'] = (data_daily['k_day'] > data_daily['d_day']) & (
                data_daily['k_day'].shift(1) < data_daily['d_day'].shift(1))
        data_daily['day_kd_s'] = (data_daily['k_day'] < data_daily['d_day']) & (
                data_daily['k_day'].shift(1) > data_daily['d_day'].shift(1))
        data_week['week_kd_b'] = (data_week['k_week'] > data_week['d_week']) & (
                data_week['k_week'].shift(1) < data_week['d_week'].shift(1))
        data_week['week_kd_s'] = (data_week['k_week'] < data_week['d_week']) & (
                data_week['k_week'].shift(1) > data_week['d_week'].shift(1))
        data_daily = data_daily.dropna().sort_values(by=['date_time'], ascending=False)
        data_week = data_week.dropna().sort_values(by=['date_time'], ascending=False)
        day_kd_s_lst = data_daily.day_kd_s.tolist()
        day_kd_b_lst = data_daily.day_kd_b.tolist()
        week_kd_s_lst = data_week.week_kd_s.tolist()
        week_kd_b_lst = data_week.week_kd_b.tolist()

        if day_kd_s_lst[0] == True:
            kd_max_now = max(data_daily.h_k_day.tolist()[0], data_daily.h_d_day.tolist()[0])
            price_max_now = data_daily.h_day.tolist()[0]
            i = 0
            for idx, _row in data_daily.iterrows():
                if _row.day_kd_s == True:
                    i += 1
                    if i == 2:
                        kd_max = max(_row.h_k_day, _row.h_d_day)
                        price_max = _row.h_day
                        if kd_max_now < kd_max and price_max_now > price_max:
                            print('%s:日级别顶背离' % index_code)
                            row = []
                            row.append(_row.stock_code)
                            row.append('日级别顶背离')
                            df_lst.append(row)
        if day_kd_b_lst[0] == True:
            kd_min_now = min(data_daily.l_k_day.tolist()[0], data_daily.l_d_day.tolist()[0])
            price_min_now = data_daily.l_day.tolist()[0]
            i = 0
            for idx, _row in data_daily.iterrows():
                if _row.day_kd_b == True:
                    i += 1
                    if i == 2:
                        kd_min = max(_row.l_k_day, _row.l_d_day)
                        price_min = _row.l_day
                        if kd_min_now > kd_min and price_min_now < price_min:
                            print('%s:日级别底背离' % index_code)
                            row = []
                            row.append(_row.stock_code)
                            row.append('日级别底背离')
                            df_lst.append(row)
        if week_kd_s_lst[0] == True:
            kd_max_now = max(data_week.h_k_week.tolist()[0], data_week.h_d_week.tolist()[0])
            price_max_now = data_week.h_week.tolist()[0]
            i = 0
            for idx, _row in data_week.iterrows():
                if _row.week_kd_s == True:
                    i += 1
                    if i == 2:
                        kd_max = max(_row.h_k_week, _row.h_d_week)
                        price_max = _row.h_week
                        if kd_max_now < kd_max and price_max_now > price_max:
                            print('%s:周级别顶背离' % index_code)
                            row = []
                            row.append(_row.stock_code)
                            row.append('周级别顶背离')
                            df_lst.append(row)
        if week_kd_b_lst[0] == True:
            kd_min_now = min(data_week.l_k_week.tolist()[0], data_week.l_d_week.tolist()[0])
            price_min_now = data_week.l_week.tolist()[0]
            i = 0
            for idx, _row in data_week.iterrows():
                if _row.week_kd_b == True:
                    i += 1
                    if i == 2:
                        kd_min = max(_row.l_k_week, _row.l_d_week)
                        price_min = _row.l_week
                        if kd_min_now > kd_min and price_min_now < price_min:
                            print('%s:周级别底背离' % index_code)
                            row = []
                            row.append(_row.stock_code)
                            row.append('周级别底背离')
                            df_lst.append(row)

    df = pd.DataFrame(df_lst, columns=['代码', '背离'])

    name_lst = []
    for code in df['代码'].tolist():
        name_lst.append(get_security_info(code).display_name)
    df['简称'] = name_lst
    df['日期'] = end_day

    df['代码'] = df['代码'].apply(lambda x: x[:-9])
    df = df[['日期', '简称', '代码', '背离']]
    df['简称'] = df['简称'].apply(lambda x: x[:-4])
    print(df)
    df = df.sort_values(['背离'])

    df_hold = df[df['代码'].isin(hold_code_lst)]
    other_code = [i for i in index_code_lst if i not in hold_code_lst]
    df_other = df[df['代码'].isin(other_code)]

    df.to_csv(fold_data + 'kdj_future_' + end_day + '.csv', encoding='gbk')
    print(time.time() - t0)
    PDFGenerator('kd_beili_' + end_day).genTaskPDF(df_hold, df_other)