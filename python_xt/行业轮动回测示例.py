# encoding:gbk
import pandas as pd
import numpy as np
from base_function import *

#myContextInfo=type('t',{},())()

def init(ContextInfo):
	#if not hasattr(myContextInfo,'get_trade_detail_data'):
	#	myContextInfo.get_trade_detail_data = get_trade_detail_data
	ContextInfo.get_trade_detail_data = get_trade_detail_data
	initialize(ContextInfo, 'AI智能驱动')
	ContextInfo.Log = []
	ContextInfo.get_trade_detail_data = []



def handlebar(ContextInfo):
	#if not hasattr(myContextInfo,'get_trade_detail_data'):
	#	myContextInfo.get_trade_detail_data = get_trade_detail_data
	ContextInfo.get_trade_detail_data = get_trade_detail_data
	if ContextInfo.is_last_bar():
		print(ContextInfo.get_datetime())
		ContextInfo.get_trade_detail_data = get_trade_detail_data
		handlebar_xt(ContextInfo, timetag_to_datetime, passorder)
	ContextInfo.Log = []
	ContextInfo.get_trade_detail_data = []
	ContextInfo.timetag_to_datetime = []
	ContextInfo.passorder = []

