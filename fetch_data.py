# !pip install git+https://github.com/pgradz/FinRL.git

# # ## install finrl library
# !pip install wrds
# !pip install swig
# !apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
# !pip install git+https://github.com/pgradz/FinRL.git

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

%matplotlib inline
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

import itertools

import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

tickers = ['XLY', 'XLF', 'XLV', 'XLE', 'XLK', 'XLI', 'XLU', 'TLT', 'GLD', 'SPY','^TNX','DX-Y.NYB','^MOVE','^VIX', 'DBC','BDRY']

fred_tickers = ['T5YIE','FEDFUNDS','M2SL','WEI','BAMLH0A0HYM2EY','GS10','GS2']

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config_tickers import DOW_30_TICKER

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-01-01'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2025-07-15' # last period is not fully utlized so extending beyond

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TEST_END_DATE,
                     ticker_list = tickers).fetch_data()


df.to_csv('sector_data.csv',index=False)

from pandas_datareader import data as pdr

# List of FRED tickers
'''If you're documenting this in your scientific work:

State clearly that indicators were shifted by their known publication lag to prevent data leakage.

Optionally cite FRED's metadata for each series (which documents publication timing).

You can say:

"To prevent forward-looking bias, all macroeconomic indicators were lag-adjusted according to their known publication delay (e.g., M2SL: 5 days, FEDFUNDS: 1 day)."
'''
fred_tickers = ['T5YIE','FEDFUNDS','M2SL','WEI','BAMLH0A0HYM2EY','GS10','GS2']
start_date = '2010-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# FRED tickers with accurate publication lags in business days
fred_series_with_lags = {
    'T5YIE': 0,   # Breakeven Inflation: available daily
    'FEDFUNDS': 1,  # Published next day
    'M2SL': 5,    # Published with ~5-day lag (weekly)
    'WEI': 1,     # Published next day
    'BAMLH0A0HYM2EY': 1,  # Published with 1-day delay
    'GS10': 0,    # Daily
    'GS2': 0      # Daily
}

# Download and shift according to publication lag
macro_data = pd.DataFrame()

for ticker, lag in fred_series_with_lags.items():
    df = pdr.DataReader(ticker, 'fred', start=start_date, end=end_date)
    df.columns = [ticker]
    if lag > 0:
        df[ticker] = df[ticker].shift(lag)
    macro_data = pd.concat([macro_data, df], axis=1)
    print(f"{ticker} downloaded and shifted by {lag} business days")

# Forward fill missing data
macro_data.ffill(inplace=True)
macro_data.dropna(inplace=True)

# Derived features
macro_data['real_fed_funds'] = macro_data['FEDFUNDS'] - macro_data['T5YIE']
macro_data['term_spread'] = macro_data['GS10'] - macro_data['GS2']
macro_data['high_yield_spread'] = macro_data['BAMLH0A0HYM2EY'] - macro_data['GS10']
macro_data['m2_growth'] = macro_data['M2SL'].pct_change().fillna(0)

# Reset index and preview
macro_data.reset_index(inplace=True)
print(macro_data.head())

macro_data.to_csv('macro_data.csv',index=False)