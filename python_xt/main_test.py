# encoding:gbk
import pandas as pd
import numpy as np
import param
from base_function import *


def init(ContextInfo):
	initialize(ContextInfo, get_trade_detail_data)
	ContextInfo.Log = []
	ContextInfo.get_trade_detail_data = []



def handlebar(ContextInfo):
	handlebar_xt(ContextInfo, get_trade_detail_data, timetag_to_datetime)
	ContextInfo.Log = []
	ContextInfo.get_trade_detail_data = []



if __name__ == '__main__':
    a = param.start_date[0:4] + param.start_date[5:7] + param.start_date[8:10]
    print(a)