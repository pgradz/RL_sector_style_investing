import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict
import datetime

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.env_portfolio_allocation.env_portfolio_sequence import StockPortfolioSequenceEnv
from finrl.agents.stablebaselines3.models import DRLAgent, CustomLSTMPolicy, CustomTransformerPolicy, CustomCNNPolicy, CustomCNNLSTMPolicy
import torch
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

from pyfolio import timeseries
sys.path.append("../FinRL-Library")

TOTAL_TIMESTEPS = 50000
policy = CustomCNNPolicy
# TODO: to use other algos, policies have to be refactor to inherit from different parent clases
'''    POLICY_BASES = {
        'ppo': ActorCriticPolicy,
        'a2c': ActorCriticPolicy,
        'ddpg': DDPGPolicy,
        'td3': TD3Policy,
        'sac': SACPolicy,
    }'''

model = 'ppo'

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

etf = pd.read_csv('datasets/sector_ETFs.csv',parse_dates=['date'])
stocks = [ 'SPY', 'TLT', 'XLE', 'XLF', 'XLI', 'XLK','XLU', 'XLV', 'XLY']
macro_indicators = ['DBC', 'DX-Y.NYB', 'GLD','^MOVE', '^TNX', '^VIX', 'BDRY']
macro = pd.read_csv('datasets/macro_data.csv',parse_dates=['DATE'])
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
unique_dates = df['date'].unique()

train = data_split(df, '2009-01-01','2021-01-01')
val = data_split(df, '2021-01-01','2022-01-01')
trade = data_split(df,'2022-01-01', '2025-01-01')

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# 4) Environment constructor function
def make_env(the_df, **kwargs):
    return StockPortfolioSequenceEnv(df=the_df, **kwargs)

env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "transaction_cost_pct": 0.001, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.INDICATORS, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4,
    "sequence_length": 20
    
}
# 7) Define a small hyperparam grid
param_grid_ppo = [
    {"learning_rate": 3e-4, "n_steps": 1024},
    {"learning_rate": 1e-4, "n_steps": 2048},
    {"ent_coef":0.01, "n_steps": 2048, "learning_rate": 0.00025,"batch_size": 128},
    {"ent_coef": 0.005}
]
param_grid_ddpg = [
    { # Variant 0: from the repo
        "buffer_size": 10_000,
        "learning_rate": 0.0005,
        "batch_size": 64
    },
 # Variant 1: Baseline / moderate
    {
        "buffer_size": 50_000,   # enough to store up to half of 100k timesteps
        "learning_rate": 3e-4,   # moderate LR
        "batch_size": 64,        # standard
        "tau": 0.005,            # soft update speed
        "gamma": 0.99,
        "learning_starts": 1_000 # how many steps before we start learning
    },
    # Variant 2: Slightly higher LR, smaller buffer
    {
        "buffer_size": 20_000,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.99,
        "learning_starts": 500
    },
    # Variant 3: Lower LR, bigger batch
    {
        "buffer_size": 50_000,
        "learning_rate": 1e-4,
        "batch_size": 256,       # larger batch
        "tau": 0.01,             # slightly faster target updates
        "gamma": 0.99,
        "learning_starts": 2_000
    },
    # Variant 4: Smaller batch, quicker learning start
    {
        "buffer_size": 20_000,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "tau": 0.005,
        "gamma": 0.99,
        "learning_starts": 200
    },
    # Variant 5: Tweak tau and gamma
    {
        "buffer_size": 30_000,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "tau": 0.02,     # more aggressive soft update
        "gamma": 0.98,   # slightly lower discount
        "learning_starts": 1_500
    }
]

def get_rl_model_params(model_name: str) -> Dict:
    """Returns parameters for the given RL model name."""
    params = {
        "ppo": param_grid_ppo,
        # "a2c": param_grid_a2c,
        # "sac": param_grid_sac,
        "ddpg": param_grid_ddpg
        # "td3": param_grid_td3
    }
    return params.get(model_name.lower(), {})

def get_policy_kwargs_grid(policy_class) -> list:
    """Returns policy kwargs grid for the given policy class."""
    policy_kwargs_grids = {
        CustomCNNLSTMPolicy: [
            {
                "cnn_filters": [32, 64],
                "cnn_kernel_sizes": [3, 3],
                "cnn_dropout": 0.1,
                "lstm_hidden_size": 64,
                "lstm_num_layers": 1,
                "lstm_dropout": 0.0,
                "net_arch": [64, 64],
                "activation_fn": torch.nn.ReLU
            }
        ],
        CustomCNNPolicy: [
            {                      # CNN-specific parameters
            "num_filters": [32, 64],
            "kernel_sizes": [3, 3],
            "dropout": 0.1,
            "net_arch": [64, 64],
            "activation_fn": torch.nn.ReLU
            }
        ],
        CustomLSTMPolicy: [
            # TODO: Add LSTM-specific policy kwargs grid
        ],
        CustomTransformerPolicy: [
            # TODO: Add Transformer-specific policy kwargs grid
        ]
    }
    return policy_kwargs_grids.get(policy_class, [{}])

e_train_gym = StockPortfolioSequenceEnv(df = train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

e_eval_gym = StockPortfolioSequenceEnv(df=val,**env_kwargs)

env_eval, _ = e_eval_gym.get_sb_env()
print(type(env_eval))

# initialize
agent = DRLAgent(env = env_train)

params = get_rl_model_params(model)
policy_kwargs = get_policy_kwargs_grid(policy)
# 8) Run search_best_hparams
best_params, best_model = agent.search_best_hparams(
    model_name=model,
    train_df=train,
    val_df=val,
    param_grid=params,
    policy = policy,
    policy_kwargs_grid = policy_kwargs,
    total_timesteps=20_000,
    env_constructor=make_env,
    eval_freq=5_000,
    best_model_save_path="./best_hparam_search",
    **env_kwargs
)


print("Hyperparam Search Results:")
print("Best Params: ", best_params)
print("We have a best_model trained with these params")

# 9) Once hyperparams are found, proceed with walk-forward:
start_date = "2020-10-01"
end_date   = "2024-12-31"

#   We'll choose a 63-day validation window each iteration, 
#   then a 63-day trading window (rebalance_window).
rebalance_window = 63
val_window       = 63

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

df_res, df_account_all, df_actions_all = agent.walk_forward_final_vs_checkpoint(
    df=df,
    unique_trade_dates=unique_dates,
    start_date=start_date,
    end_date=end_date,
    model_name=model,
    fixed_params=best_params,   # from search_best_hparams
    rebalance_window=rebalance_window,
    val_window=val_window,
    total_timesteps=50_000,
    env_constructor=make_env,
    eval_freq=5000,
    best_model_prefix="./walkforward_best_model",
    seed=seed,
    **env_kwargs
)


print("Walk-Forward Results Summary:")
print(df_res)
df_res.to_csv(f'./results/walkforward_results_{model}_{seed}.csv')
print("Account Value Memory Over All Trading Windows:")
print(df_account_all.head())
df_account_all.to_csv(f'./results/walkforward_account_value_{model}_{seed}.csv')
print("Actions Memory Over All Trading Windows:")
print(df_actions_all.head())
df_actions_all.to_csv(f'./results/walkforward_actions_{model}_{seed}.csv')
perf_stats_all = backtest_stats(account_value=df_account_all)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv(f'./results/walkforward_perf_stats_{model}_{seed}.csv')




# # model with CNN + LSTM policy
# model_ppo_cnn_lstm = agent.get_model(
#     model_name="ppo",                    # or "a2c", "sac", etc.
#     policy=CustomCNNLSTMPolicy,          # Pass the class directly
#     policy_kwargs={                      # CNN + LSTM-specific parameters
#         "cnn_filters": [32, 64],
#         "cnn_kernel_sizes": [3, 3],
#         "cnn_dropout": 0.1,
#         "lstm_hidden_size": 64,
#         "lstm_num_layers": 1,
#         "lstm_dropout": 0.0,
#         "net_arch": [64, 64],
#         "activation_fn": torch.nn.ReLU
#     },
#     verbose=1
# )
# trained_ppo_cnn_lstm = agent.train_model(model=model_ppo_cnn_lstm,
#                                 tb_log_name='ppo_cnn_lstm',
#                                 total_timesteps=TOTAL_TIMESTEPS)

# e_trade_gym = StockPortfolioSequenceEnv(df = trade, **env_kwargs)
# df_daily_return_cnn_lstm, df_actions_cnn_lstm, _ = DRLAgent.DRL_prediction(model=trained_ppo_cnn_lstm,
#                         environment = e_trade_gym)  

# # model with CNN policy
# model_ppo_cnn = agent.get_model(
#     model_name="ppo",                    # or "a2c", "sac", etc.
#     policy=CustomCNNPolicy,              # Pass the class directly
#     policy_kwargs={                      # CNN-specific parameters
#         "num_filters": [32, 64],
#         "kernel_sizes": [3, 3],
#         "dropout": 0.1,
#         "net_arch": [64, 64],
#         "activation_fn": torch.nn.ReLU
#     },
#     verbose=1
# )
# trained_ppo_cnn = agent.train_model(model=model_ppo_cnn,
#                                 tb_log_name='ppo_cnn',
#                                 total_timesteps=TOTAL_TIMESTEPS)

# e_trade_gym = StockPortfolioSequenceEnv(df = trade, **env_kwargs)
# df_daily_return_cnn, df_actions_cnn , _ = DRLAgent.DRL_prediction(model=trained_ppo_cnn,
#                         environment = e_trade_gym)

# # model with LSTM policy
# model_ppo_lstm = agent.get_model(
#     model_name="ppo",                    # or "a2c", "sac", etc.
#     policy=CustomLSTMPolicy,             # Pass the class directly
#     policy_kwargs={                      # LSTM-specific parameters
#         "lstm_hidden_size": 128,
#         "lstm_num_layers": 2,
#         "lstm_dropout": 0.1,
#         "net_arch": [64, 64],
#         "activation_fn": torch.nn.ReLU
#     },
#     verbose=1
# )

# trained_ppo_lstm = agent.train_model(model=model_ppo_lstm, 
#                                 tb_log_name='ppo_lstm',
#                                 total_timesteps=TOTAL_TIMESTEPS)

# e_trade_gym = StockPortfolioSequenceEnv(df = trade, **env_kwargs)
# df_daily_return_lstm, df_actions_lstm, _ = DRLAgent.DRL_prediction(model=trained_ppo_lstm,
#                         environment = e_trade_gym)

# # model with Transformer policy
# model_ppo_transformer = agent.get_model(
#     model_name="ppo",                    # or "a2c", "sac", etc.
#     policy=CustomTransformerPolicy,      # Pass the class directly
#     policy_kwargs={                      # Transformer-specific parameters
#         "num_layers": 2,
#         "num_heads": 8,
#         "embed_dim": 128,
#         "dropout": 0.1,
#         "net_arch": [64, 64],
#         "activation_fn": torch.nn.ReLU
#     },
#     verbose=1
# )

# trained_ppo_transformer = agent.train_model(model=model_ppo_transformer,
#                                 tb_log_name='ppo_transformer',
#                                 total_timesteps=TOTAL_TIMESTEPS)      
# e_trade_gym = StockPortfolioSequenceEnv(df = trade, **env_kwargs)
# df_daily_return_transformer, df_actions_transformer, _ = DRLAgent.DRL_prediction(model=trained_ppo_transformer,
#                         environment = e_trade_gym) 



# DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return_lstm)
# perf_func = timeseries.perf_stats 
# perf_stats_all = perf_func( returns=DRL_strat, 
#                               factor_returns=DRL_strat, 
#                                 positions=None, transactions=None, turnover_denom="AGB")
# print('Performance Statistics for LSTM Policy: ')
# print(perf_stats_all)

# perf_stats_all = perf_func( returns=convert_daily_return_to_pyfolio_ts(df_daily_return_transformer),
#                               factor_returns=convert_daily_return_to_pyfolio_ts(df_daily_return_transformer),
#                                 positions=None, transactions=None, turnover_denom="AGB")
# print('Performance Statistics for Transformer Policy: ')
# print(perf_stats_all)   

# perf_stats_all = perf_func( returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn),
#                                 factor_returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn),
#                                 positions=None, transactions=None, turnover_denom="AGB")
# print('Performance Statistics for CNN Policy: ')

# print(perf_stats_all)   
# perf_stats_all = perf_func( returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn_lstm),
#                                 factor_returns=convert_daily_return_to_pyfolio_ts(df_daily_return_cnn_lstm),
#                                 positions=None, transactions=None, turnover_denom="AGB")
# print('Performance Statistics for CNN + LSTM Policy: ')
# print(perf_stats_all)