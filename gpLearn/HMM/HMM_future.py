# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/3 10:34
@Author  : zhangfang
@File    : HMM_future.py
'''
import quandl
import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
from sklearn.externals import joblib
import pandas as pd
import joblib
from backtest_func import yearsharpRatio, maxRetrace, annROR
from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, date_rules, time_rules, get_datetime)

YOUR_QUANDL_API = 'UbykF8XfAjaNx1zXQYpD'

# 特征工程和建模
def get_best_hmm_model(X, max_states, max_iter=10000):
    best_score = -(10**10)
    best_state = 0

    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components=state, random_state=100, covariance_type='diag',
                                n_iter=max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state
    best_model = GaussianHMM(n_components=best_state, random_state=100, covariance_type='diag', n_iter=max_iter).fit(X)
    return best_model


# Normalizde
def std_normallized(vals):
    return np.std(vals) / np.mean(vals)


# Ratio of diff
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]


# z-score
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


def plot_hidden_states(model, data, X, column_price):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(model.n_components, 3, figsize=(15, 15))
    colours = cm.prism(np.linspace(0, 1, model.n_components))
    hidden_states = model.predict(X)
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(data.index, data[column_price], c='grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c=colour)
        ax[0].set_title('{0}th hidder state'.format(i))
        ax[0].grid(True)

        ax[1].hist(data['future_return'][mask], bins=30)
        ax[1].set_xlim([-0.1, 0.1])
        ax[1].set_title('future return distrbution at {0}th hidder state'.format(i))
        ax[1].grid(True)

        ax[2].plot(data['future_return'][mask].cumsum(), c=colour)
        ax[2].set_title('cummulative future return at {0}th hidden state'.format(i))
        ax[2].grid(True)
    plt.tight_layout()


# 第二个关键 是通过特征来研究每个状态。在此之后，我们可以将这两个事件（未来走势和当前状态）联系起来。让我们为每个状态的特征编写代码和可视化。
def mean_confidence_interval(vals, confidence):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m, m+h


def compare_hidden_states(hmm_model, cols_features, conf_interval, iters=1000):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(len(cols_features), hmm_model.n_components, figsize=(15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    for i in range(0, hmm_model.n_components):
        mc_df = pd.DataFrame()

        # Samples generation
        for j in range(0, iters):
            row = np.transpose(hmm_model._generate_sample_from_state(i))
            mc_df = mc_df.append(pd.DataFrame(row).T)
        mc_df.columns = cols_features
        for k in range(0, len(mc_df.columns)):
            axs[k][i].hist(mc_df[cols_features[k]], color=colours[i])
            axs[k][i].set_title(cols_features[k] + ' (state ' + str(i) + '): ' +
                                str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)
    plt.tight_layout()


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def initialize(context):
    context.asset = symbol('btc_usd')
    context.leverage = 1.0
    context.std_period = 10
    context.ma_period = 10
    context.price_deviation_period = 10
    context.volume_deviation_period = 10
    context.n_periods = 5 + int(np.max([context.std_period, context.ma_period, context.price_deviation_period,
                                        context.volume_deviation_period]))
    context.tf = '1440T'
    context.model = joblib.load('quandl_BITFINEX_BTCUSD_final_model.pkl')
    context.cols_fetures = ['last_return', 'std_normalized', 'ma_ratio', 'price_deviation', 'volume_deviation']
    context.long_states = [2]
    context.random_states = [1]
    context.short_states = [0]

    context.set_commission(maker=0.002, taker=0.002)
    context.set_slippage(slippage=0.0005)
    context.set_benchmark(context.asset)


def handle_data(context, data):
    current_date = get_datetime().date()
    current_time = get_datetime().time()

    if current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0:
        prices = pd.DataFrame()
        volumes = pd.DataFrame()
        try:
            prices = data.history(context.asset, fields='price', bar_count=context.n_periods, frequency=context.tf)
            volumes = data.history(context.asset, fields='volume', bar_count=context.n_periods, frequency=context.tf)
        except:
            print('NO DATA')
        if prices.shape[0] == context.n_periods and volumes.shape[0] == context.n_periods:
            features = pd.DataFrame()
            features['price'] = prices
            features['volume'] = volumes
            features['last_return'] = features['price'].pct_change()
            features['std_normalized'] = features['price'].rolling(context.std_period).apply(std_normallized)
            features['ma_ratio'] = features['price'].rolling(context.ma_period).apply(ma_ratio)
            features['price_deviation'] = features['price'].rolling(context.price_deviation_period).apply(values_deviation)
            features['volume_deviation'] = features['volume'].rolling(context.volume_deviation_period).apply(
                values_deviation)
            state = context.random_states[0]
            if features.dropna().shape[0] == context.n_periods - context.ma_period + 1:
                state = int(context.model.predict(features[context.cols_fetures].dropna())[-1])
            else:
                print('PROBLEM:features dataframe is too small')
            print('State on ' + str(current_date) + ' ' + str(current_time) + ': ' + str(state))
            print('Amount on ' + str(current_date) + ' ' + str(current_time) + ': ' + str(context.portfolio.positions[context.asset].amount))
            print(prices.dropna())
            print(volumes.dropna())
            if context.portfolio.positions[context.asset].amount <= 0 and state in context.long_states:
                print('LONG on ' + str(current_date) + ' ' + str(current_time))
                order_target_percent(context.asset, 1.0 * context.leverage)
                context.best_price_ts = data.current(context.asset, 'close')
            if context.portfolio.positions[context.asset].amount != 0 and state in context.random_states:
                print('CLOSE on ' + str(current_date) + ' ' + str(current_time))
                order_target_percent(context.asset, 0)
            if context.portfolio.positions[context.asset].amount >= 0 and state in context.short_states:
                print('SHORT on ' + str(current_date) + ' ' + str(current_time))
                order_target_percent(context.asset, -1.0 * context.leverage)
                context.best_price_ts = data.current(context.asset, 'close')
            record(price=prices[-1], state=state, amount=context.portfolio.positions[context.asset].amount)


def analyze(context, perf):
    sns.set()
    # Summary output
    print('Total return: ' + str(perf.algorithm_period_return[-1]))
    print('Sortina coef: ' + str(perf.sortino[-1]))
    print('Max drawdown: ' + str(np.min(perf.max_drawdown[-1])))
    print('alpha: ' + str(perf.alpha[-1]))
    print('beta: ' + str(perf.beta[-1]))
    f = plt.figure(figsize=(7.2, 7.2))

    # Plot return
    ax1 = f.add_subplot(211)
    ax1.plot(perf.algorithm_period_return, 'blue')
    ax1.plot(perf.benchmark_period_return, 'red')
    ax1.legend()
    ax1.set_title('Returns')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')

    # Plot state
    ax2 = f.add_subplot(212, sharex=ax1)
    ax2.plot(perf.state, 'grey')
    ax2.set_title('state')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_date_string = '2014-04-01'
    asset = 'BITFINEX/BTCUSD'
    column_price = 'Last'
    column_high = 'High'
    column_low = 'Low'
    column_volume = 'Volume'
    # 数据来自quandl
    quandl.ApiConfig.api_key = YOUR_QUANDL_API
    dataset = quandl.get(asset, collapse='daily', trim_start=start_date_string)
    dataset = dataset.shift(1)

    # 训练集：01 / 01 / 2018 之前。下面的代码有关特征工程：

    # Feature params
    future_period = 1
    std_period = 10
    ma_period = 10
    price_deviation_period = 10
    volume_deviation_period = 10

    # Create features
    cols_features = ['last_return', 'std_normalized', 'ma_ratio', 'price_deviation', 'volume_deviation']
    dataset['last_return'] = dataset[column_price].pct_change()
    dataset['std_normalized'] = dataset[column_price].rolling(std_period).apply(std_normallized)
    dataset['ma_ratio'] = dataset[column_price].rolling(ma_period).apply(ma_ratio)
    dataset['price_deviation'] = dataset[column_price].rolling(price_deviation_period).apply(values_deviation)
    dataset['volume_deviation'] = dataset[column_volume].rolling(volume_deviation_period).apply(values_deviation)

    dataset['future_return'] = dataset[column_price].pct_change(future_period).shift(-future_period)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    # Split the data on sets
    train_ind = int(np.where(dataset.index == '2018-01-01 00:00:00')[0])
    train_set = dataset[cols_features].values[:train_ind]
    test_set = dataset[cols_features].values[train_ind:]

    # Plot features
    plt.figure(figsize=(20, 10))
    fig, axs = plt.subplots(len(cols_features), 1, figsize=(15, 15))
    colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
    for i in range(0, len(cols_features)):
        axs[i].plot(dataset.reset_index()[cols_features[i]], color=colours[i])
        axs[i].set_title(cols_features[i])
        axs[i].grid(True)

    plt.tight_layout()
    # 然后我们得到了五个新的时间序列和训练模型：
    model = get_best_hmm_model(train_set, 3)
    plot_hidden_states(model, dataset[:train_ind].reset_index(), train_set, column_price)

    # 正如我们看到的，状态  # 0有下降的趋势。状态#1没有一个明确的趋势。最后一个状态#2有强烈的上行趋势。 这个带有累积和future_return的简单技巧使我们能够理解每个状态如何对应下一个价格波动。

    # 第二个关键 是通过特征来研究每个状态。在此之后，我们可以将这两个事件（未来走势和当前状态）联系起来。让我们为每个状态的特征编写代码和可视化。
    compare_hidden_states(hmm_model=model, cols_features=cols_features, conf_interval=0.95)

    # 搭建策略
    # 状态为#0时： 做空
    # 状态为#1时： 空仓
    # 状态为#2时： 做多

    #使用    Catalyst    框架

    run_algorithm(capital_base=10000,
                  data_frequency='minute',
                  initialize=initialize,
                  handle_data=handle_data,
                  analyze=analyze,
                  exchange_name='bitfinex',
                  quote_currency='usd',
                  start=pd.to_datetime('2018-1-1', utc=True),
                  end=pd.to_datetime('2019-5-22', utc=True))

