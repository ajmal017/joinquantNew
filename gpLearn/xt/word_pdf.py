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
        smtp = smtplib.SMTP()
        smtp.connect('smtp.163.com')
        smtp.login(username, password)
        smtp.sendmail(sender, receiver, msg.as_string())
        smtp.quit()
        print('完成邮件发送')


if __name__ == "__main__":
    FormatConvert.word_to_pdf("C:\\e\\data\\qe\\simulate\\report\\模拟交易报告20201012.docx", "C:\\e\\data\\qe\\simulate\\report\\模拟交易报告202010123.pdf")

