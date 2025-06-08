#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:40:21 2025

BIAS:https://www.oanda.com/bvi-ft/lab-education/technical_analysis/bias/
RSI:https://www.oanda.com/bvi-ft/lab-education/technical_analysis/what_is_rsi/


@author: ggyuen & Hugh
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


stockName=input("Stock Name:")
hk1810 = yf.Ticker(stockName)
#stock = hk1810.history(start='2025-03-28',end='2025-03-29', interval='1m')
stock = hk1810.history(period='1d', interval='1m')#28/3/2024
stock.index = stock.index.tz_localize(None)
pd.options.display.max_columns = None

#log returns
stock['log_return'] = np.log(stock['Close'] / stock['Close'].shift(1))
stock.dropna(inplace=True)

#moving averages
stock['30min'] = stock['Close'].rolling(window=30).mean()  # 30-minute MA
stock.dropna(subset=['30min'], inplace=True)

#BIAS using 30-min MA
stock['BIAS'] = (stock['Close'] - stock['30min']) / stock['30min'] * 100

stock.dropna(subset=['BIAS'], inplace=True)

#RSI
delta = stock['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock['RSI'] = 100 - (100 / (1 + rs))

#logic
stock['Position1'] = np.where(stock['RSI'] >= 80, -1, np.where(stock['RSI'] <= 30, 1, 0))

stock['Position2'] = np.where(stock['BIAS'] > 0, 1, -1)

stock['Position'] = np.where(stock['Position1'] == 1, 1,  
                             np.where(stock['Position1'] == -1, -1,  
                             np.where(stock['Position2'] == 1, -1,  
                             np.where(stock['Position2'] == -1, 1, 
                             0))))  

#strategy returns
stock['Strategy'] = stock['Position'].shift(1) * stock['log_return']

print(stock)

#test
def calculate_metrics(stock):
    stock['Daily_Return'] = stock['Close'].pct_change()
    #total strategy return
    total_strategy_return = stock['Strategy'].sum()
    #variance of daily returns
    variance_daily_returns = stock['Daily_Return'].var()
    #standard deviation of strategy returns
    std_strategy_return = stock['Strategy'].std()
    #annualized return
    annualized_return = (1 + total_strategy_return) ** (252 / len(stock)) - 1
    # Prepare results
    metrics = {
        'Total Strategy Return': total_strategy_return,
        'Variance of Daily Returns': variance_daily_returns,
        'Standard Deviation of Strategy Returns': std_strategy_return,
        'Annualized Return': annualized_return,
    }

    return metrics

sampleValue = calculate_metrics(stock)
for key, value in sampleValue.items():
    print(f"{key}: {value:.10f}")

# Print results
print(stock[['log_return', 'Strategy']].sum())
print(stock[['log_return', 'Strategy']].std() * 252 ** 0.5)


buy_signals = stock[stock['Position'] == 1]
sell_signals = stock[stock['Position'] == -1]
plt.show()

#singalplot
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', alpha=1,)
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', alpha=1)

# Plot
plt.plot(stock.index, stock['Close'], label='Close Price', color='black')
plt.plot(stock.index, stock['30min'], label='30-min MA', color='yellow')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.title(stockName+' Analysis')
plt.show()

#test
#stock[['RSI','Position']].plot(style='-')
#stock['BIAS'].plot(secondary_y='BIAS', style='-')
#plt.show()

#backtest
stock[['log_return','Strategy']].cumsum().plot()
stock['Position'].plot(secondary_y='Position', style='--')
plt.title('Back testing of '+stockName+' using BIAS&RSI')
plt.show()
