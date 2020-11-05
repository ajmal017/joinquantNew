#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
pip install comtypes -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
"""

import os
import smtplib
from comtypes import client
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from win32com.client import Dispatch
from email_fuction import get_html_msg

def get_html_msg(data):
    """
    1. 构造html信息
    """
    df = data.copy()
    df_html = df.to_html(escape=False)

    head = \
        "<head> \
            <meta charset='utf-8'> \
            <STYLE TYPE='text/css' MEDIA=screen> \
                table.dataframe { \
                    border-collapse: collapse;\
                    border: 2px solid \
                    /*居中显示整个表格*/ \
                    margin: auto; \
                } \
                table.dataframe thead { \
                    border: 2px solid #91c6e1;\
                    background: #f1f1f1;\
                    padding: 10px 10px 10px 10px;\
                    color: #333333;\
                }\
                table.dataframe tbody {\
                    border: 2px solid #91c6e1;\
                    padding: 10px 10px 10px 10px;\
                }\
                table.dataframe tr { \
                } \
                table.dataframe th { \
                    vertical-align: top;\
                    font-size: 14px;\
                    padding: 10px 10px 10px 10px;\
                    color: #105de3;\
                    font-family: arial;\
                    text-align: center;\
                }\
                table.dataframe td { \
                    text-align: center;\
                    padding: 10px 10px 10px 10px;\
                }\
                body {\
                    font-family: 宋体;\
                }\
                h1 { \
                    color: #5db446\
                }\
                div.header h2 {\
                    color: #0002e3;\
                    font-family: 黑体;\
                }\
                div.content h2 {\
                    text-align: center;\
                    font-size: 28px;\
                    text-shadow: 2px 2px 1px #de4040;\
                    color: #fff;\
                    font-weight: bold;\
                    background-color: #008eb7;\
                    line-height: 1.5;\
                    margin: 20px 0;\
                    box-shadow: 10px 10px 5px #888888;\
                    border-radius: 5px;\
                }\
                h3 {\
                    font-size: 22px;\
                    background-color: rgba(0, 2, 227, 0.71);\
                    text-shadow: 2px 2px 1px #de4040;\
                    color: rgba(239, 241, 234, 0.99);\
                    line-height: 1.5;\
                }\
                h4 {\
                    color: #e10092;\
                    font-family: 楷体;\
                    font-size: 20px;\
                    text-align: center;\
                }\
                td img {\
                    /*width: 60px;*/\
                    max-width: 300px;\
                    max-height: 300px;\
                }\
            </STYLE>\
        </head>\
        "
    # 构造模板的附件（100）
    body = "<body>\
        <div align='center' class='header'> \
            <!--标题部分的信息-->\
            <h1 align='center'>内容 </h1>\
        </div>\
        <hr>\
        <div class='content'>\
            <!--正文内容-->\
            <h2>内容：</h2>\
            <div>\
                <h4></h4>\
                {df_html}\
            </div>\
            <hr>\
            <p style='text-align: center'>\
                —— 本次报告完 ——\
            </p>\
        </div>\
        </body>\
        ".format(df_html=df_html)
    html_msg = "<html>" + head + body + "</html>"
    # 这里是将HTML文件输出，作为测试的时候，查看格式用的，正式脚本中可以注释掉
    # fout = open('t4.html', 'w', encoding='UTF-8', newline='')
    # fout.write(html_msg)
    return html_msg


class FormatConvert(object):

    @staticmethod
    def word_to_pdf(word_path, pdf_path, type="offie"):
        if not os.path.exists(word_path):
            raise FileNotFoundError('%s 文件不存在' % word_path)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            # raise FileExistsError('%s 文件已存在' % pdf_path)

        if type == "offie":
            word = Dispatch('Word.Application')
            doc = word.Documents.Open(word_path)
            doc.SaveAs(pdf_path, FileFormat=17)
            doc.Close()
            word.Quit()
        elif type == "wps":
            word = client.CreateObject("Word.Application")
            word.Visible = 0
            obj = word.Documents.Open(word_path)
            obj.SaveAs(pdf_path, FileFormat=17)
            obj.Close()


class SendMessage(object):
    def __init__(self, sender, password, server="smtp.exmail.qq.com", port=465):
        """
        :param sender:
        :param password:
        :param server:
        :param port:
        """
        self.sender = sender
        self.password = password
        self.server = server
        self.port = port

    def send_email(self, subject, content, receivers, file_path=None):
        """
        发送邮件
        :param subject:
        :param content:
        :param receivers:
        :param file_path:
        :return:
        """
        message = MIMEMultipart()
        message.attach(MIMEText(content, "plain", "utf-8"))
        message['Subject'] = subject
        message['To'] = ";".join(receivers)
        message['From'] = self.sender
        if file_path:
            filename = file_path.split('/')[-1]
            att = MIMEText(open(file_path, 'rb').read(), 'base64', 'utf-8')
            att.add_header("Content-Disposition", 'attachment', filename=Header(filename, 'utf-8').encode())
            message.attach(att)
        smtp = smtplib.SMTP_SSL(self.server, self.port)
        smtp.login(self.sender, self.password)
        smtp.sendmail(self.sender, receivers, message.as_string())

    def send_email1(self, resualt, subject, From, receiver):
        # 通过Header对象编码的文本，包含utf-8编码信息和Base64编码信息。以下中文名测试ok
        # subject = '中文标题'
        # subject=Header(subject, 'utf-8').encode()

        # 构造邮件对象MIMEMultipart对象
        # 下面的主题，发件人，收件人，日期是显示在邮件页面上的。
        msg = MIMEMultipart('mixed')
        msg['Subject'] = subject
        msg['From'] = From
        # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
        msg['To'] = ";".join(receiver)
        # msg['Date']='2012-3-16'

        # 构造文字内容
        text = subject
        text_plain = MIMEText(text, 'plain', 'utf-8')
        msg.attach(text_plain)
        html_msg = get_html_msg(resualt)
        content_html = MIMEText(html_msg, "html", "utf-8")
        msg.attach(content_html)

        # 发送邮件
        smtp = smtplib.SMTP_SSL(self.server, self.port)
        # smtp.connect('smtp.163.com')
        smtp.login(self.sender, self.password)
        smtp.sendmail(self.sender, receiver, msg.as_string())
        smtp.quit()
        print('完成邮件发送')


if __name__ == "__main__":
    FormatConvert.word_to_pdf("C:\\e\\data\\qe\\simulate\\report\\模拟交易报告20201012.docx", "C:\\e\\data\\qe\\simulate\\report\\模拟交易报告202010123.pdf")

