# coding=utf-8
import docx
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.shared import Inches
from docx.enum.table import WD_TABLE_ALIGNMENT
# from DBM.mongo_handle import MongoHandle
import pandas as pd
# from global_config import *
import datetime


def make_doc(self, today=None):
    """
    生成报表
    :return:
    """
    today = datetime.date.today() if not today else pd.to_datetime(today)
    last_trade_date = basic_data.find_tradeday(-1, today)
    bfr_monday_date = basic_data.find_tradeday(-1, today + datetime.timedelta(days=-today.weekday(), weeks=0))
    bfr_month_date = basic_data.find_tradeday(-1, today.replace(day=1))
    bfr_year_date = basic_data.find_tradeday(-1, today.replace(month=1, day=1))
    today = today.strftime('%Y%m%d')
    fund_list = list()
    chat_paths = list()
    for name, obj in self.hosts.items():
        chat_path = obj.draw_chat(today)  # 画图
        fund = obj.fund_now.copy()  # 资金
        fund_his = obj.fund_his.copy().append(fund)
        fund = fund_his.loc[fund_his['交易日'] == today]  # 资金
        key = fund_his['交易日'] >= last_trade_date
        fund['当日收益'] = f"{round((fund.iloc[-1]['净值'] / fund_his.loc[key, '净值'].iloc[0] - 1) * 100, 2)}%"
        key = fund_his['交易日'] >= bfr_monday_date
        fund['当周收益'] = f"{round((fund.iloc[-1]['净值'] / fund_his.loc[key, '净值'].iloc[0] - 1) * 100 ,2)}%"
        key = fund_his['交易日'] >= bfr_month_date
        fund['当月收益'] = f"{round((fund.iloc[-1]['净值'] / fund_his.loc[key, '净值'].iloc[0] - 1) * 100, 2)}%"
        key = fund_his['交易日'] >= bfr_year_date
        fund['当年收益'] = f"{round((fund.iloc[-1]['净值'] / fund_his.loc[key, '净值'].iloc[0] - 1) * 100, 2)}%"
        fund = fund[['策略名称', '资产总值', '证券市值', '总盈亏', '净值份额', '净值', '当日收益', '当周收益',
                     '当月收益', '当年收益']]
        fund[['资产总值', '证券市值', '总盈亏']] = round(fund.iloc[-1]['资产总值'], 2), \
                                        round(fund.iloc[-1]['证券市值'], 2), round(fund.iloc[-1]['总盈亏'], 2)
        fund['净值'] = round(fund.iloc[-1]['净值'], 4)
        fund_list.append(fund)
        chat_paths.append(chat_path)
    funds = pd.concat(fund_list)
    funds.sort_values('策略名称', inplace=True)
    document = Document()
    # 添加标题，格式2
    for section in document.sections:
        section.page_width = Inches(10)
    document.add_heading()


def save_df_to_doc(document, test_df, word=None):
    """
    将结果按照dataframe的形式存入doc文件
    :param document: 存入的文档类
    :param test_df: 需要保存的df
    :return:

    """
    document.styles['Normal'].font.name = u'宋体'
    document.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')

    # add_paragraph表示添加一个段落
    if word:
        document.add_paragraph(u'\n%s' % (word))

    # 添加一个表格--行数和列数，行数多加一行，需要将列名同时保存
    t = document.add_table(test_df.shape[0] + 1, test_df.shape[1], style="Table Grid")
    t.alignment = WD_TABLE_ALIGNMENT.CENTER  # 表格整体居中
    # 将每列列名保存到表格中
    for j in range(test_df.shape[-1]):
        t.cell(0, j).text = test_df.columns[j]
        t.cell(0, j).width = Inches(1.85)

    # 将每列数据保存到新建的表格中
    for i in range(test_df.shape[0]):
        for j in range(test_df.shape[-1]):
            # 第一行保存的是列名，所以数据保存时，行数要加1
            t.cell(i + 1, j).text = str(test_df.values[i, j])

    for row in t.rows:
        for cell in row.cells:
            paragraphs = cell.paragraphs
            for paragraph in paragraphs:
                for run in paragraph.runs:
                    font = run.font
                    font.size = Pt(7.5)
    for col in range(test_df.shape[1]):
        t.cell(0, col).width = Inches(1.65)


if __name__ == '__main__':
    date = '20200710'
    path = 'G:/trading/trading_report/'
    account1 = '21900576'
    account2 = '85030120'
    host_name = 'account_0710113519'
    document = Document()
    document.add_picture(f'{path}/account_{date}.png', width=Inches(6.0))
    document.add_picture(f'{path}/JZ_ZG_{date}.png', width=Inches(6.0))
    document.add_picture(f'{path}/JZ_ROE_{date}.png', width=Inches(6.0))


    with MongoHandle.mongo_connect() as cl:
        fund_df = pd.DataFrame(cl['hosts'][host_name + '_fund'].find({'交易日': date}, {'_id': 0}))
        fund_df = fund_df[['策略名称', '交易日', '资产总值', '证券市值', '可用余额', '总盈亏', '净值份额', '净值']]
        fund_df[['资产总值', '证券市值', '可用余额', '总盈亏']] = \
            fund_df[['资产总值', '证券市值', '可用余额', '总盈亏']].round(2)
        fund_df['净值'] = fund_df['净值'].round(4)
        save_df_to_doc(document, fund_df, '策略概要')
        deal_df = pd.DataFrame(cl['hosts'][host_name + '_deal'].find({'交易日': date}, {'_id': 0}))
        if not deal_df.empty:
            deal_df = deal_df[['证券代码', '证券名称', '交易类型', '成交价格', '成交数量', '成交时间',
                               '成交金额', '手续费', '策略名称']]
            deal_df['手续费'] = deal_df['手续费'].round(2)
        else:
            deal_df = pd.DataFrame(columns=['证券代码', '证券名称', '交易类型',
                                            '成交价格', '成交数量', '成交时间', '成交金额', '手续费', '策略名称'])
        save_df_to_doc(document, deal_df, '交易统计')
        hold_df = pd.DataFrame(cl['hosts'][host_name + '_hold'].find({'交易日': date}, {'_id': 0}))
        hold_df = hold_df[['证券代码', '证券名称', '当前拥股数量', '冻结数量', '可卖数量', '成本价格', '市场价',
                           '浮动盈亏', '盈亏比例', '证券市值']]
        hold_df.rename(columns={'当前拥股数量': '余股', '冻结数量': '冻结', '可卖数量': '可卖'}, inplace=True)
        hold_df['成本价格'] = hold_df['成本价格'].round(2)
        hold_df['浮动盈亏'] = hold_df['浮动盈亏'].round(2)
        save_df_to_doc(document, hold_df, '期末持仓')
        # document.add_paragraph(u'/n')
        # document.add_picture('E:/OneDrive/Workspace/股票/交易报告/预期策略模拟盘交易报告/1.png', width=Inches(6.0))
        # document.add_picture('E:/OneDrive/Workspace/股票/交易报告/预期策略模拟盘交易报告/2.png', width=Inches(6.0))
        # document.add_picture('E:/OneDrive/Workspace/股票/交易报告/预期策略模拟盘交易报告/3.png', width=Inches(6.0))
        document.save(f'{path}/{host_name}策略交易报告{date}.docx')
        print('报告生成完成')