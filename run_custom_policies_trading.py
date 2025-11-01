import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import datetime

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent, CustomLSTMPolicy, CustomTransformerPolicy, CustomCNNPolicy, CustomCNNLSTMPolicy
import torch
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from pyfolio import timeseries
sys.path.append("../FinRL-Library")


import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


train = pd.read_csv('datasets/train_data.csv',parse_dates=['date'])
trade = pd.read_csv('datasets/trade_data.csv',parse_dates=['date'])
df = pd.concat([train,trade])
# keep only date	open	high	low	close	adjcp	volume	tic	day
df = df[['date','tic','open','high','low','close','volume']]

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)


# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
# lookback=252
# for i in range(lookback,len(df.index.unique())):
#   data_lookback = df.loc[i-lookback:i,:]
#   price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
#   return_lookback = price_lookback.pct_change().dropna()
#   return_list.append(return_lookback)

#   covs = return_lookback.cov().values 
#   cov_list.append(covs)

  
# df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
# df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)
from finrl.config import INDICATORS


train = data_split(df, '2009-01-01','2021-01-01')
val = data_split(df, '2021-01-01','2022-01-01')
trade = data_split(df,'2022-01-01', '2025-01-01')

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
# explanation in https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py starts in line 407
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# initialize
agent = DRLAgent(env = env_train)


model_a2c = agent.get_model("a2c")
trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=50000)

# model with LSTM policy
model_ppo_lstm = agent.get_model(
    model_name="ppo",                    # or "a2c", "sac", etc.
    policy=CustomLSTMPolicy,             # Pass the class directly
    policy_kwargs={                      # LSTM-specific parameters
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.1,
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    },
    verbose=1
)

trained_ppo_lstm = agent.train_model(model=model_ppo_lstm, 
                                tb_log_name='ppo_lstm',
                                total_timesteps=10000)

e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_lstm, df_actions_lstm, _ = DRLAgent.DRL_prediction(model=trained_ppo_lstm,
                        environment = e_trade_gym)




# model with Transformer policy
model_ppo_transformer = agent.get_model(
    model_name="ppo",                    # or "a2c", "sac", etc.
    policy=CustomTransformerPolicy,      # Pass the class directly
    policy_kwargs={                      # Transformer-specific parameters
        "num_layers": 2,
        "num_heads": 8,
        "embed_dim": 128,
        "dropout": 0.1,
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    },
    verbose=1
)

trained_ppo_transformer = agent.train_model(model=model_ppo_transformer,
                                tb_log_name='ppo_transformer',
                                total_timesteps=10000)      
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_transformer, df_actions_transformer, _ = DRLAgent.DRL_prediction(model=trained_ppo_transformer,
                        environment = e_trade_gym) 

# model with CNN policy
model_ppo_cnn = agent.get_model(
    model_name="ppo",                    # or "a2c", "sac", etc.
    policy=CustomCNNPolicy,              # Pass the class directly
    policy_kwargs={                      # CNN-specific parameters
        "num_filters": [32, 64],
        "kernel_sizes": [3, 3],
        "dropout": 0.1,
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    },
    verbose=1
)
trained_ppo_cnn = agent.train_model(model=model_ppo_cnn,
                                tb_log_name='ppo_cnn',
                                total_timesteps=10000)

e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_cnn, df_actions_cnn , _ = DRLAgent.DRL_prediction(model=trained_ppo_cnn,
                        environment = e_trade_gym)

# model with CNN + LSTM policy
model_ppo_cnn_lstm = agent.get_model(
    model_name="ppo",                    # or "a2c", "sac", etc.
    policy=CustomCNNLSTMPolicy,          # Pass the class directly
    policy_kwargs={                      # CNN + LSTM-specific parameters
        "cnn_filters": [32, 64],
        "cnn_kernel_sizes": [3, 3],
        "cnn_dropout": 0.1,
        "lstm_hidden_size": 64,
        "lstm_num_layers": 1,
        "lstm_dropout": 0.0,
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    },
    verbose=1
)
trained_ppo_cnn_lstm = agent.train_model(model=model_ppo_cnn_lstm,
                                tb_log_name='ppo_cnn_lstm',
                                total_timesteps=10000)

e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_cnn_lstm, df_actions_cnn_lstm, _ = DRLAgent.DRL_prediction(model=trained_ppo_cnn_lstm,
                        environment = e_trade_gym)  


DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return_lstm)
perf_func = timeseries.perf_stats 
perf_stats_all = perf_func( returns=DRL_strat, 
                              factor_returns=DRL_strat, 
                                positions=None, transactions=None, turnover_denom="AGB")
print('Performance Statistics for LSTM Policy: ')
print(perf_stats_all)

perf_stats_all = perf_func( returns=convert_daily_return_to_pyfolio_ts(df_daily_return_transformer),
                              factor_returns=convert_daily_return_to_pyfolio_ts(df_daily_return_transformer),
                                positions=None, transactions=None, turnover_denom="AGB")
print('Performance Statistics for Transformer Policy: ')
print(perf_stats_all)   

perf_stats_all = perf_func( returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn),
                                factor_returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn),
                                positions=None, transactions=None, turnover_denom="AGB")
print('Performance Statistics for CNN Policy: ')

print(perf_stats_all)   
perf_stats_all = perf_func( returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn_lstm),
                                factor_returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn_lstm),
                                positions=None, transactions=None, turnover_denom="AGB")
print('Performance Statistics for CNN + LSTM Policy: ')
print(perf_stats_all)