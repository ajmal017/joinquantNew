from functools import wraps
import copy
import traceback

class __PyContext(object):
    def __init__(self, contextinfo=None):
        self.context = contextinfo
        self.z8sglma_last_version = None
        self.z8sglma_last_barpos = -1
    
    def set_account(self, acct):
        self.context.set_account(acct)

    def set_universe(self, universe):
        last_universe = self.context.get_universe()
        universe = list(set(universe).difference(set(last_universe)))
        self.context.set_universe(universe)

    def get_universe(self):
        return self.context.get_universe()
    
    def is_last_bar(self):
        return self.context.is_last_bar()

    def is_new_bar(self):
        return self.context.is_new_bar()
    
    def get_history_data(self, len, period, field, dividend_type='none', skip_paused=True):
        return self.context.get_history_data(len, period, field, dividend_type, skip_paused)
    
    def get_industry(self, industry_name, real_timetag = -1):
        return self.context.get_industry(industry_name, real_timetag)

    def get_last_close(self,stock):
        return self.context.get_last_close(stock)

    def get_last_volume(self,stock):
        return self.context.get_last_volume(stock)
    
    def get_sector(self, sectorname, real_timetag = -1):
        return self.context.get_sector(sectorname, real_timetag)

    def get_scale_and_stock(self, total, stockValue, stock):
        return self.context.get_scale_and_stock(total, stockValue, stock)
    def get_scale_and_rank(self,list):
        return self.context.get_scale_and_rank(list)
		
    def get_finance(self, vStock):
        return self.context.get_finance(vStock)
		
    def get_smallcap(self):
        return self.context.get_smallcap()

    def get_midcap(self):
        return self.context.get_midcap()
    
    def get_largecap(self):
        return self.context.get_largecap()
        
    def get_bar_timetag(self, index):
        return self.context.get_bar_timetag(index)

    def get_tick_timetag(self):
        return self.context.get_tick_timetag()

    def get_risk_free_rate(self, index):
        return self.context.get_risk_free_rate(index)

    def get_contract_multiplier(self, stockcode):
        return self.context.get_contract_multiplier(stockcode)
    
    def get_float_caps(self, stockcode):
        return self.context.get_float_caps(stockcode)
    
    def get_total_share(self, stockcode):
        return self.context.get_total_share(stockcode)
    
    def get_stock_type(self, stock):
        return self.context.get_stock_type(stock)

    def get_stock_name(self, stock):
        return self.context.get_stock_name(stock)

    def get_open_date(self, stock):
        return self.context.get_open_date(stock)

    def get_svol(self, stock):
        return self.context.get_svol(stock)

    def get_bvol(self, stock):
        return self.context.get_bvol(stock)

    def get_net_value(self, barpositon):
        return self.context.get_net_value(barpositon)
    
    def get_back_test_index(self):
        return self.context.get_back_test_index()
    
    def get_turn_over_rate(self, stockcode):
        return self.context.get_turn_over_rate(stockcode) 
    
    def get_weight_in_index(self, mtkindexcode, stockcode):
        return self.context.get_weight_in_index(mtkindexcode, stockcode)
    
    def get_stock_list_in_sector(self, sectorname, real_timetag = -1):
        return self.context.get_stock_list_in_sector(sectorname, real_timetag)
        
    def get_tradedatafromerds(self, accounttype, accountid, startdate, enddate):
        return self.context.get_tradedatafromerds(accounttype ,accountid, startdate, enddate)

    def get_close_price(self, market, stockCode, realTimetag, period=86400000, dividType=0):
        return self.context.get_close_price(market, stockCode, realTimetag, period, dividType)

    def get_market_data(self, fields, stock_code=[], start_time='', end_time='', skip_paused=True, period='follow', dividend_type='follow', count=-1):
        oriData=self.context.get_market_data(fields, stock_code, start_time, end_time, skip_paused, period, dividend_type, count)
        resultDict={}
        for code in oriData:
            for timenode in oriData[code]:
                values=[]
                for field in fields:
                    values.append(oriData[code][timenode][field])
                key=code+timenode
                resultDict[key]=values
        if len(fields)==1 and len(stock_code)<=1 and ((start_time=='' and end_time=='') or start_time==end_time) and count==-1:
            for key in resultDict:
                return resultDict[key][0]
            return -1
        import numpy as np
        import pandas as pd
        if len(stock_code)<=1 and start_time=='' and end_time=='' and count==-1:
            for key in resultDict:
                result=pd.Series(resultDict[key],index=fields)
                return result.sort_index()
        if len(stock_code)>1 and start_time=='' and end_time=='' and count==-1:
            values=[]
            for code in stock_code:
                if code in oriData:
                    if not oriData[code]:
                        values.append([np.nan])
                    for timenode in oriData[code]:
                        key=code+timenode
                        values.append(resultDict[key])
                else:
                    values.append([np.nan])
            result=pd.DataFrame(values,index=stock_code,columns=fields)
            return result.sort_index()
        if len(stock_code)<=1 and ((start_time!='' or end_time!='') or count>=0):
            values=[]
            times=[]
            for code in oriData:
                for timenode in oriData[code]:
                    key=code+timenode
                    times.append(timenode)
                    values.append(resultDict[key])
            result=pd.DataFrame(values,index=times,columns=fields)
            return result.sort_index()
        if len(stock_code)>1 and ((start_time!='' or end_time!='') or count>=0):
            values={}
            for code in stock_code:
                times=[]
                value=[]
                if code in oriData:
                    for timenode in oriData[code]:
                        key=code+timenode
                        times.append(timenode)
                        value.append(resultDict[key])
                values[code]=pd.DataFrame(value,index=times,columns=fields).sort_index()
            result=pd.Panel(values)
            return result
        return

    def get_full_tick(self, stock_code=[]):
        return self.context.get_full_tick(stock_code)
    def load_stk_list(self, dirfile, namefile):
        return self.context.load_stk_list(dirfile, namefile)
    def load_stk_vol_list(self, dirfile, namefile):
        return self.context.load_stk_vol_list(dirfile, namefile)
    def get_longhubang(self, stock_list =[], startTime = '', endTime = '', count = -1):
        import types;
        import pandas as pd;
        resultDf =  pd.DataFrame();
        if type(endTime) is int :
            count = endTime;
            endTime = startTime;
            startTime = '0';
        else:
            count = -1;
        resultDict = self.context.get_longhubang(stock_list,startTime,endTime,count);
        fields = ['stockCode','stockName','date','reason','close','SpreadRate','TurnoverVolune','Turnover_Amount',"buyTraderBooth","sellTraderBooth"];
        tradeBoothItemFiled = ["traderName","buyAmount","buyPercent","sellAmount","sellPercent","totalAmount","rank","direction"];
        for stock in resultDict:
            stockDict = resultDict[stock];
            stockDf = pd.DataFrame();
            if len(list(stockDict.keys())) < 10:
                continue;
            buyTradeBoothDict = stockDict[8];
            sellTradeBoothDict = stockDict[9];
            buyTradeBoothPdList = [];
            sellTradeBoothPdList = [];
            for TradeBoothIDict in buyTradeBoothDict:
                buyTradeBoothPd = pd.DataFrame();
                for tradeBoothKey in list(TradeBoothIDict.keys()):
                    buyTradeBoothPd[tradeBoothItemFiled[tradeBoothKey]] = TradeBoothIDict[tradeBoothKey];
                buyTradeBoothPdList.append(buyTradeBoothPd);
            for TradeBoothIDict in sellTradeBoothDict:
                sellTradeBoothPd = pd.DataFrame();
                for tradeBoothKey in list(TradeBoothIDict.keys()):
                    sellTradeBoothPd[tradeBoothItemFiled[tradeBoothKey]] = TradeBoothIDict[tradeBoothKey];
                sellTradeBoothPdList.append(sellTradeBoothPd);
            for i in range(0,8):
                stockDf[fields[i]] = stockDict[i];
            stockDf[fields[8]] = buyTradeBoothPdList;
            stockDf[fields[9]] = sellTradeBoothPdList;
            resultDf = resultDf.append(stockDf);
        return resultDf;

    def get_main_contract(self, codemarket):
        return self.context.get_main_contract(codemarket)

    def get_date_location(self, date):
        return self.context.get_date_location(date)
    
    def get_product_share(self, code, index=-1):
        return self.context.get_product_share(code, index)

    def get_divid_factors(self, marketAndStock, date = ''):
        return self.context.get_divid_factors(marketAndStock,date)

    def get_financial_data(self, fieldList, stockList, startDate, endDate, report_type = 'announce_time', pos = -1):
        if(type(report_type) != str):   # default value error , report_type -> pos 
            pos = report_type;
            report_type = 'announce_time';
        if(report_type != 'announce_time' and report_type != 'report_time'):
            return;
        if type(fieldList) == str and type(stockList) == str:
            return self.context.get_financial_data(fieldList, stockList, startDate, endDate, report_type, pos);
        import pandas as pd
        from collections import OrderedDict
        
        pandasData = self.context.get_financial_data(fieldList, stockList, startDate, endDate,report_type)
        if not pandasData:
            return
        fields = pandasData['field']
        stocks = pandasData['stock']
        dates = pandasData['date']
        values = pandasData['value']

        if len(stocks) == 1 and len(dates) == 1:    #series
            series_list = []
            for value in values:
                if not value:
                    return
                for subValue in value:
                    series_list.append(subValue)
            return pd.Series(series_list, index = fields)
        elif len(stocks) == 1 and len(dates) > 1:   #index = dates, col = fields
            dataDict = OrderedDict()
            for n in range(len(values)):
                dataList = []
                if not values[n]:
                    return
                for subValue in values[n]:
                    dataList.append(subValue)
                dataDict[fields[n]] = pd.Series(dataList, index = dates)
            return pd.DataFrame(dataDict)
        elif len(stocks) > 1 and len(dates) == 1:   #index = stocks col = fields
            dataDict = OrderedDict()
            for n in range(len(values)):
                dataList = []
                if not values[n]:
                    return
                for subValue in values[n]:
                    dataList.append(subValue)
                dataDict[fields[n]] = pd.Series(dataList, index = stocks)
            return pd.DataFrame(dataDict)
        else:                                       #item = stocks major = dates minor = fields
            panels = OrderedDict()
            for i in range(len(stocks)):
                dataDict = OrderedDict()
                for j in range(len(values)):
                    dataList = []
                    value = values[j]
                    if not value:
                        return
                    for k in range(i * len(dates), (i + 1) * len(dates)):
                        dataList.append(value[k])
                    dataDict[fields[j]] = pd.Series(dataList, index = dates)
                panels[stocks[i]] = pd.DataFrame(dataDict)
            return pd.Panel(panels)
            
    def get_top10_share_holder(self, stock_list, data_name,start_time,end_time):
        import pandas as pd;
        resultPanelDict = {};
        resultDict ={};
        if(data_name == 'flow_holder' or data_name == 'holder'):
            resultDict = get_top10_holder(stock_list, data_name, start_time, end_time);
        else:
            return "input data_name = \'flow_holder\' or data_name = \'holder\'";
        fields = ["holdName","holderType","holdNum","changReason","holdRatio","stockType","rank","status","changNum","changeRatio"];
        for stock in resultDict:
            stockPdData = pd.DataFrame(columns = fields);
            stockDict = resultDict[stock];
            for timeKey in list(stockDict.keys()):
                timelist = stockDict[timeKey];
                stockPdData.loc[timeKey] =  timelist;
            resultPanelDict[stock] = stockPdData;
        resultPanel = pd.Panel(resultPanelDict);
        stockNum = len(stock_list);
        timeNum = len(resultPanel.major_axis);
        if(stockNum == 1 and timeNum == 1):
            stock = resultPanel.items[0];
            timetag = resultPanel.major_axis[0];
            df = pd.DataFrame(resultPanel[stock]);
            result = pd.Series(df.ix[timetag],index = fields);
            return result;
        elif(stockNum > 1 and timeNum == 1):
            timetag = resultPanel.major_axis[0];
            result = pd.DataFrame(resultPanel.major_xs(timetag),index =  fields,columns = resultPanel.items);
            result = result.T;
            return  result;
        elif(stockNum == 1 and timeNum > 1):
            stock = resultPanel.items[0];
            result = pd.DataFrame(resultPanel[stock]);
            return result;
        elif(stockNum > 1 and timeNum > 1):
            return resultPanel;
        return pd.Panel();
    def get_product_asset_value(self, code, index=-1):
        return self.context.get_product_asset_value(code, index)

    def get_product_init_share(self,code=''):
        return self.context.get_product_init_share(code)

    def create_sector(self, sectorname, stocklist):
        return self.context.create_sector(sectorname, stocklist)    
    def get_holder_num(self, stock_list =[], startTime = '', endTime = ''):
        fields = ["stockCode","timetag","holdNum","AHoldNum","BHoldNum","HHoldNum","uncirculatedHoldNum","circulatedHoldNum"];
        import pandas as pd;
        resultDict = get_holder_number(stock_list, startTime, endTime)
        result  =  pd.DataFrame();
        for stock in resultDict:
            df = pd.DataFrame(columns = fields);
            for i in  resultDict[stock]:
                df[fields[i]]=resultDict[stock][i];
            result = result.append(df);
        return  result;
        
    def paint(self, name, data, index, drawStyle, selectcolor='', limit=''):
        selectcolor_low = selectcolor.lower()
        limit_low = limit.lower()
        if '' != selectcolor and 'noaxis' == limit_low:
            return self.context.paint(name, data, index, drawStyle, selectcolor, 0)
        elif '' != selectcolor and 'nodraw' == limit_low:
            return self.context.paint(name, data, index, 7, selectcolor, 0)
        elif 'noaxis' == selectcolor_low:
            return self.context.paint(name, data, index, drawStyle, '', 0)
        elif 'nodraw' == selectcolor_low:
            return self.context.paint(name, data, index, 7, '', 0)
        else:
            return self.context.paint(name, data, index, drawStyle, selectcolor_low, 1)

    def set_slippage(self, b_flag, slippage='none'):
        if slippage != 'none':
            self.context.set_slippage(b_flag,slippage)
        else:
            self.context.set_slippage(b_flag)#b_flag=slippage
            
    
    def get_slippage(self):
        return self.context.get_slippage()
    
    def get_commission(self):
        return self.context.get_commission()

    def set_commission(self,comtype,com='none'):
        if com != 'none':
           self.context.set_commission(comtype,com)
        else:
           self.context.set_commission(0,comtype)#comtype=commission
    
    def is_suspended_stock(self, stock):
        return self.context.is_suspended_stock(stock)    

    def is_stock(self,stock):
        return self.context.is_stock(stock)

    def is_fund(self,stock):
        return self.context.is_fund(stock)

    def is_future(self,market):
        return self.context.is_future(market)

    def run_time(self,funcname, intervalday, time, exchange):
        self.context.run_time(funcname, intervalday, time, exchange)

    def get_function_line(self):
        import sys
        return sys._getframe().f_back.f_lineno
        
    def get_trading_dates(self, stockcode, start_date, end_date, count, period='1d'):
        return self.context.get_trading_dates(stockcode, start_date, end_date, count, period)

    def draw_text(self, condition, position, text, limit=''):
        import sys
        line = sys._getframe().f_back.f_lineno
        if 'noaxis' == limit.lower():
            return self.context.draw_text(condition, position, text, line, 0)
        else:
            return self.context.draw_text(condition, position, text, line, 1)

    def draw_vertline(self, condition, price1, price2, color='', limit=''):
        import sys
        line = sys._getframe().f_back.f_lineno

        if 'noaxis' == limit.lower():
            return self.context.draw_vertline(condition, price1, price2, color, line, 0)
        else:
            return self.context.draw_vertline(condition, price1, price2, color, line, 1)

    def draw_icon(self, condition, position, type, limit=''):
        import sys
        line = sys._getframe().f_back.f_lineno
        if (('noaxis' == limit.lower())):
            return self.context.draw_icon(condition, position, type, line, 0)
        else:
            return self.context.draw_icon(condition, position, type, line, 1)

    def draw_number(self, cond, price, number, precision, limit=''):
        import sys
        line = sys._getframe().f_back.f_lineno
        if (('noaxis' == limit.lower())):
            return self.context.draw_number(cond, price, number, precision, line, 0)
        else:
            return self.context.draw_number(cond, price, number, precision, line, 1)
        
    def get_turnover_rate(self, stock_code=[], start_time='19720101', end_time='22010101'):
        import pandas as pd
        import time
        if(len(start_time) != 8 or len(end_time) != 8):
            print('input date time error!!!')
            return pd.DataFrame()
        data = turnover_rate(stock_code, start_time, end_time)
        frame = pd.DataFrame(data)
        
        return frame;
        
    def get_ETF_list(self, market, stockcode, typeList = []):
        import pandas as pd
        if(len(market) == 0):
            print('input market error!!!')
            return pd.DataFrame()
        data = get_etf_list(market, stockcode, typeList)
        frame = pd.DataFrame(data)
        
        return data;
        
    def get_option_detail_data(self, stockcode):
        return self.context.get_option_detail_data(stockcode)
        
    def get_instrumentdetail(self, marketCode):
        field_list = [
            'ExchangeID'
            , 'InstrumentID'
            , 'InstrumentName'
            , 'ProductID'
            , 'ProductName'
            , 'CreateDate'
            , 'OpenDate'
            , 'ExpireDate'
            , 'PreClose'
            , 'SettlementPrice'
            , 'UpStopPrice'
            , 'DownStopPrice'
            , 'FloatVolumn'
            , 'TotalVolumn'
            , 'LongMarginRatio'
            , 'ShortMarginRatio'
            , 'PriceTick'
            , 'VolumeMultiple'
            , 'MainContract'
            , 'LastVolume'
            , 'InstrumentStatus'
            , 'IsTrading'
            , 'IsRecent'
        ]

        inst = self.context.get_instrumentdetail(marketCode)

        ret = {}
        for field in field_list:
            ret[field] = inst.get(field)
        return ret

    @property
    def time_tick_size(self):
        return self.context.time_tick_size
    
    @property
    def current_bar(self):
        return self.context.current_bar
    
    @property
    def barpos(self):
        return self.context.barpos

    @property
    def benchmark(self):
        return self.context.benchmark

    @benchmark.setter
    def benchmark(self, value):
        self.context.benchmark = value
    
    @property
    def period(self):
        return self.context.period
    
    @property
    def capital(self):
        return self.context.capital

    @property
    def dividend_type(self):
        return self.context.dividend_type

    @capital.setter
    def capital(self, value):
        self.context.capital = value
    
    @property
    def refresh_rate(self):
        return self.context.refresh_rate

    @refresh_rate.setter
    def refresh_rate(self, value):
        self.context.refresh_rate = value
        
    @property
    def do_back_test(self):
        return self.context.do_back_test

    @do_back_test.setter
    def do_back_test(self, value):
        self.context.do_back_test = value
        
    @property
    def request_id(self):
        return self.context.request_id

        
    @property
    def stockcode(self):
        return self.context.stockcode
    @property
    def stockcode_in_rzrk(self):
        return self.context.stockcode_in_rzrk
    
    @property
    def market(self):
        return self.context.market
        
    @property
    def in_pythonworker(self):
        return self.context.in_pythonworker

    @property
    def start(self):
        return self.context.start
        
    @start.setter
    def start(self,value):
       self.context.start = value

    @property
    def end(self):
        return self.context.end
        
    @end.setter
    def end(self,value):
       self.context.end = value

    @property
    def data_info_level(self):
        return self.context.data_info_level
        
    @data_info_level.setter
    def data_info_level(self,value):
       self.context.data_info_level = value   

    def __deepcopy__(self, memo):
        #print "type:", type(self)
        new_obj = type(self)()
        # del last version when copy, only the last version is reverved
        # self.z8sglma_last_version = None
        for k, v in list(self.__dict__.items()):
            #print "k: %s v: %s" %(k, v)
            # contextInfo variable is from c++, not copy
            if k == "context":
                setattr(new_obj, k, v)
            elif k == "z8sglma_last_version":
                continue
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj

def timetag_to_datetime(timetag, format):
    import time
    timetag = timetag/1000
    time_local = time.localtime(timetag)
    return time.strftime(format,time_local)


def resume_context_info(context_info):
    last_barpos = context_info.z8sglma_last_barpos
    if context_info.barpos == last_barpos:
        for k, v in list(context_info.z8sglma_last_version.__dict__.items()):
            if k == "context":
                continue
            elif k == "z8sglma_last_version":
                continue
            else:
                setattr(context_info, k, copy.deepcopy(v))
    else:
        # print "not repeat, barpos:", args[0].barpos
        # print "curr bar: %i last bar: %i" % (args[0].barpos, context_info.last_barpos)
        context_info.z8sglma_last_barpos = context_info.barpos
        context_info.z8sglma_last_version = copy.deepcopy(context_info)

