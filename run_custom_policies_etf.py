import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict
import datetime
import os
import json

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.env_portfolio_allocation.env_portfolio_sequence import StockPortfolioSequenceEnv
from finrl.meta.env_stock_trading.env_stocktrading_sequence import StockTradingSequenceEnv
# Import the new MLP environment for benchmarking
from env_portfolio_mlp import StockPortfolioMLPEnv
from finrl.agents.stablebaselines3.models import (
    DRLAgent, 
    get_cnn_policy,
    get_lstm_policy,
    get_transformer_policy,
    get_cnn_lstm_policy,
    PolicyRegistry
)
import torch
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

from pyfolio import timeseries
sys.path.append("../FinRL-Library")

TOTAL_TIMESTEPS = 100000  # TODO: increase to 100k for final runs after testing

# ============================================================================
# FLEXIBLE POLICY SELECTION USING REGISTRY PATTERN
# ============================================================================
# Now you can easily switch between different RL algorithms and feature extractors!
# 
# Supported Models: 'ppo', 'a2c', 'ddpg', 'td3', 'sac'
# Supported Feature Extractors: 'CustomCNN', 'CustomLSTM', 'CustomTransformer', 'CustomCNNLSTM', 'MLP'
#
# Examples:
#   - CNN with PPO:         model='ppo',  feature_extractor='CustomCNN'
#   - LSTM with DDPG:       model='ddpg', feature_extractor='CustomLSTM'
#   - Transformer with SAC: model='sac',  feature_extractor='CustomTransformer'
#   - CNN-LSTM with TD3:    model='td3',  feature_extractor='CustomCNNLSTM'
#   - MLP with PPO:         model='ppo',  feature_extractor='MLP'  # Benchmark!
# ============================================================================

# ============================================================================
# ENVIRONMENT SELECTION: Sequence Models vs Standard MLP
# ============================================================================
# Set USE_SEQUENCE_ENV to control which environment type to use:
#   - True:  Use sequence environments (LSTM, CNN, Transformer, CNN-LSTM)
#   - False: Use standard MLP environment (single-timestep baseline)
#
# This allows fair benchmarking between temporal and non-temporal models!
# ============================================================================

USE_SEQUENCE_ENV = True  # Set to False for MLP baseline benchmark

model = 'ppo'  # Choose: 'ppo', 'a2c', 'ddpg', 'td3', 'sac'

if USE_SEQUENCE_ENV:
    feature_extractor = 'CustomCNN'  # Choose: 'CustomCNN', 'CustomLSTM', 'CustomTransformer', 'CustomCNNLSTM'
    policy = PolicyRegistry.get_policy(feature_extractor, model)
    print(f"\n{'='*60}")
    print(f"🔬 SEQUENCE MODEL MODE")
    print(f"Using policy: {policy.__name__}")
    print(f"Model: {model.upper()}, Feature Extractor: {feature_extractor}")
    print(f"Environment: Sequence-aware (temporal patterns)")
    print(f"{'='*60}\n")
else:
    feature_extractor = 'MLP'  # Standard MLP for baseline
    policy = "MlpPolicy"  # Use SB3's built-in MLP policy
    print(f"\n{'='*60}")
    print(f"📊 MLP BASELINE MODE")
    print(f"Using policy: MlpPolicy (Standard SB3)")
    print(f"Model: {model.upper()}, Feature Extractor: {feature_extractor}")
    print(f"Environment: Single-timestep (spatial patterns only)")
    print(f"{'='*60}\n")

ENV_TYPE = 'portfolio'  # NEW: Choose 'portfolio' or 'trading'

def create_results_subfolder(model_name: str, feature_extractor: str, env_type: str, 
                             reward_type: str, sequence_length: int) -> str:
    """
    Create and return a results subfolder path based on configuration.
    
    Args:
        model_name: Name of the RL algorithm (e.g., 'ppo', 'ddpg')
        feature_extractor: Feature extractor name (e.g., 'CustomCNN', 'MLP')
        env_type: Environment type ('portfolio' or 'trading')
        reward_type: Reward type (e.g., 'dsr', 'pnl')
        sequence_length: Sequence length for temporal models (or 1 for MLP)
        
    Returns:
        Path to the results subfolder
    """
    # Add prefix to distinguish sequence vs MLP environments
    env_prefix = "seq" if USE_SEQUENCE_ENV else "mlp"
    
    # Create descriptive folder name
    folder_name = f"{env_prefix}_{model_name}_{feature_extractor}_{env_type}_reward_{reward_type}_seq{sequence_length}"
    results_path = os.path.join(config.RESULTS_DIR, folder_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    return results_path

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

# train = pd.read_csv('datasets/train_data.csv',parse_dates=['date'])
# trade = pd.read_csv('datasets/trade_data.csv',parse_dates=['date'])
# df = pd.concat([train,trade])

etf = pd.read_csv('datasets/sector_data_adj_close.csv',parse_dates=['date'])
stocks = [ 'SPY', 'TLT', 'BIL', 'XLE', 'XLF', 'XLI', 'XLK','XLU', 'XLV', 'XLY', 'XLB', 'XLP']
df = etf.loc[etf['tic'].isin(stocks)]
macro_indicators = ['DBC', 'DX-Y.NYB', 'GLD','^MOVE', '^TNX', '^VIX'] # 'BDRY' removed because of missing

macro_fred = pd.read_csv('datasets/macro_data.csv',parse_dates=['DATE']) # TODO: add those to macro df
macro_df = etf.loc[etf['tic'].isin(macro_indicators)]
# transpose so tic goes into columns, close is value and date is index
# TODO: there are missing values for some columns
macro_df = macro_df.pivot(index='date', columns='tic', values='close')
macro_df.reset_index(inplace=True)
# join macro_df with macro_fred
macro_df = macro_df.merge(macro_fred, how='left', left_on='date', right_on='DATE')
#drop DATE
macro_df.drop(columns=['DATE'], inplace=True)
macro_df.ffill(inplace=True)
macro_df = macro_df.loc[macro_df.date >= '2010-06-01']


# keep only date	open	high	low	close	adjcp	volume	tic	day
df = df[['date','tic','open','high','low','close','volume']]

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)

# ---------------------------------------------------------------------------
# Stationary Technical Indicators (pragmatic: transform FinRL defaults)
# ---------------------------------------------------------------------------
# FinRL's defaults include 4 non-stationary price-level features
# (boll_ub, boll_lb, close_30_sma, close_60_sma) and MACD which scales
# with price level.  Transform them into stationary, cross-asset-comparable
# features while keeping rsi_30 and dx_30 as-is (already bounded/stationary).
for tic in df['tic'].unique():
    mask = df['tic'] == tic
    close = df.loc[mask, 'close']

    # Bollinger Band position: where close sits within the bands ∈ [0, 1]
    bb_range = df.loc[mask, 'boll_ub'] - df.loc[mask, 'boll_lb']
    df.loc[mask, 'bb_position'] = (
        (close - df.loc[mask, 'boll_lb']) / bb_range.replace(0, np.nan)
    )

    # SMA disparity: % deviation from 30-day SMA (mean-reverting)
    sma30 = df.loc[mask, 'close_30_sma']
    df.loc[mask, 'sma_disparity'] = (close - sma30) / sma30.replace(0, np.nan)

    # ATR ratio: 14-day ATR / close (normalised realised volatility)
    high = df.loc[mask, 'high']
    low = df.loc[mask, 'low']
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df.loc[mask, 'atr_ratio'] = tr.rolling(14).mean() / close

# Fill NaNs from rolling calculations with sensible defaults
df['bb_position']   = df['bb_position'].fillna(0.5)
df['sma_disparity'] = df['sma_disparity'].fillna(0.0)
df['atr_ratio']     = df['atr_ratio'].fillna(0.0)

# Final indicator set: 6 stationary features
CUSTOM_INDICATORS = ['rsi_30', 'dx_30', 'macd', 'bb_position', 'sma_disparity', 'atr_ratio']

# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

df = df.sort_values(['date','tic']).reset_index(drop=True)
unique_dates = df['date'].unique()
df = df.loc[df.date >= '2010-06-01'] # fred starts June 2010

train = data_split(df, '2010-06-01','2021-01-01') # fred starts June 2010
val = data_split(df, '2021-01-01','2022-01-01')
trade = data_split(df,'2022-01-01', '2026-01-01')

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# 4) Environment constructor function
# def make_env(the_df, **kwargs):
#     return StockPortfolioSequenceEnv(df=the_df, **kwargs)

# 4) Environment constructor function
def make_env(the_df, env_type='portfolio', **kwargs):
    """
    Create environment based on type selection and USE_SEQUENCE_ENV flag.
    
    Args:
        the_df: DataFrame with stock data
        env_type: 'portfolio' or 'trading'
        **kwargs: Environment-specific kwargs
        
    Returns:
        Environment instance (sequence or MLP based on USE_SEQUENCE_ENV)
    """
    if USE_SEQUENCE_ENV:
        # Use sequence-aware environments
        if env_type == 'portfolio':
            return StockPortfolioSequenceEnv(df=the_df, **kwargs)
        elif env_type == 'trading':
            return StockTradingSequenceEnv(df=the_df, **kwargs)
    else:
        # Use standard single-timestep environment
        if env_type == 'portfolio':
            # Remove sequence-specific kwargs for MLP environment
            mlp_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['sequence_length', 'flatten_observations']}
            return StockPortfolioMLPEnv(df=the_df, **mlp_kwargs)
        # Add standard trading env if needed
    
    raise ValueError(f"Unknown environment type: {env_type}")

if ENV_TYPE == 'portfolio':
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": CUSTOM_INDICATORS, 
        "action_space": stock_dimension, 
        "reward_scaling": 1,
        "macro_df": macro_df,
        "reward_type": "dsr",  #  options: "log_return", "pnl", "dsr" (differential sharpe ratio), active_return
        "reward_transform": "ewma_zscore",  # unified scaling across reward types
        "reward_beta": 0.01,
        "reward_clip": 5.0,
        "rebalancing_threshold": 0.1,  # Execution layer: 5% blocks noise trades, allows meaningful tilts
        "turnover_penalty_threshold": 0.30,  # Not used (coeff=0)
        "turnover_penalty_coeff": 0.0,  # DISABLED: Let policy learn sector bets from market feedback only
        "action_mode": "residual",  # Zero action = hold current weights = zero turnover = zero TC (no tuning needed)
        "decision_interval": 5,  # Weekly trading: act every 5 days, hold between decisions (reduces TC structurally)
        "randomize_interval_offset": True,  # Random phase offset in training for diversity
    }
    
    # Add sequence-specific kwargs only if using sequence environment
    if USE_SEQUENCE_ENV:
        env_kwargs["sequence_length"] = 20
        env_kwargs["flatten_observations"] = False  # Keep 2D for sequence models
        env_kwargs["random_start"] = True  # RC6: Training diversity via random episode starts
elif ENV_TYPE == 'trading':
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,  # Required for trading env
        "buy_cost_pct": [0.001] * stock_dimension,  # Required for trading env
        "sell_cost_pct": [0.001] * stock_dimension, # Required for trading env
        "stock_dim": stock_dimension,
        "state_space": stock_dimension,  # Required for trading env - it is not used. consider eliminating
        "tech_indicator_list": CUSTOM_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1, # change for pnl to 1e-4,
        "macro_df": macro_df,
        "reward_type": "dsr",  # Use Differential Sharpe Ratio reward
        "sharpe_window": 20       # Optional: Adjust the rolling window
    }
    
    # Add sequence-specific kwargs only if using sequence environment
    if USE_SEQUENCE_ENV:
        env_kwargs["sequence_length"] = 20
    
# 7) Define a small hyperparam grid
param_grid_ppo = [
    # 1. High exploration (RC4: prevent policy collapse)
    {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 4,           # RC4: fewer epochs to prevent over-fitting each rollout
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.05,         # RC4: strong entropy to maintain exploration
        "clip_range": 0.1,        # RC4: tighter clipping for stable updates
        "max_grad_norm": 0.5,     # RC4: gradient clipping
        "vf_coef": 0.5,
    },
    
    # 2. Maximum entropy exploration
    {
        "learning_rate": 2e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.10,         # RC4: very high entropy for anti-collapse
        "clip_range": 0.15,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
    },
    
    # 3. Conservative exploration
    {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 3,            # RC4: minimal epochs
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "ent_coef": 0.03,
        "clip_range": 0.1,
        "max_grad_norm": 0.3,
        "vf_coef": 0.5,
    },
    
    # 4. Fast adaptation with strong exploration
    {
        "learning_rate": 5e-4,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.98,
        "gae_lambda": 0.90,
        "ent_coef": 0.05,
        "clip_range": 0.1,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
    },
    
    # 5. Large batch with moderate exploration
    {
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.05,
        "clip_range": 0.15,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
    }
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
param_grid_td3 = [
    # Variant 0: Conservative baseline - stable learning
    {
        "buffer_size": 100_000,      # Large buffer for 14 years of data
        "learning_rate": 3e-4,       # Standard learning rate
        "batch_size": 256,           # Larger batch for stability
        "tau": 0.005,                # Slow target network updates
        "gamma": 0.99,               # Standard discount factor
        "policy_delay": 2,           # TD3's delayed policy updates (key feature)
        "target_policy_noise": 0.2,  # TD3's target policy smoothing
        "target_noise_clip": 0.5,    # Clip noise for stability
        "learning_starts": 5_000     # Learn after collecting diverse experiences
    },
    
    # Variant 1: Aggressive exploration - for discovering new strategies
    {
        "buffer_size": 200_000,      # Very large buffer for long-term patterns
        "learning_rate": 5e-4,       # Higher learning rate
        "batch_size": 128,           # Smaller batch for more frequent updates
        "tau": 0.01,                 # Faster target updates
        "gamma": 0.99,
        "policy_delay": 2,
        "target_policy_noise": 0.3,  # More noise for exploration
        "target_noise_clip": 0.6,
        "learning_starts": 2_000     # Start learning earlier
    },
    
    # Variant 2: Large batch, stable updates - for high-quality gradients
    {
        "buffer_size": 150_000,
        "learning_rate": 1e-4,       # Lower LR for stability
        "batch_size": 512,           # Very large batch
        "tau": 0.005,
        "gamma": 0.99,
        "policy_delay": 3,           # More delayed updates for stability
        "target_policy_noise": 0.15, # Less noise with large batches
        "target_noise_clip": 0.4,
        "learning_starts": 10_000    # Collect more data before learning
    },
    
    # Variant 3: High-frequency trading focus - quick adaptation
    {
        "buffer_size": 50_000,       # Smaller buffer for recent patterns
        "learning_rate": 3e-4,
        "batch_size": 64,            # Small batch for quick updates
        "tau": 0.02,                 # Fast target network updates
        "gamma": 0.98,               # Slightly lower discount (short-term focus)
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "learning_starts": 1_000
    },
    
    # Variant 4: Sharpe ratio optimized - financial metrics focus
    {
        "buffer_size": 100_000,
        "learning_rate": 2e-4,       # Moderate learning rate
        "batch_size": 256,
        "tau": 0.008,                # Balanced target updates
        "gamma": 0.995,              # Higher discount for long-term rewards
        "policy_delay": 2,
        "target_policy_noise": 0.1,  # Low noise for consistent policies
        "target_noise_clip": 0.3,
        "learning_starts": 7_000     # Balanced warmup period
    }
]

def get_rl_model_params(model_name: str) -> Dict:
    """Returns parameters for the given RL model name."""
    params = {
        "ppo": param_grid_ppo,
        # "a2c": param_grid_a2c,
        # "sac": param_grid_sac,
        "ddpg": param_grid_ddpg,
        "td3": param_grid_td3
    }
    return params.get(model_name.lower(), {})

def get_policy_kwargs_grid(policy_class, model_name: str) -> list:
    """
    Returns policy kwargs grid for the given policy class and model.
    
    Args:
        policy_class: The policy class (from PolicyRegistry) or string "MlpPolicy"
        model_name: Name of the RL algorithm (e.g., 'ppo', 'ddpg')
        
    Returns:
        List of policy kwargs dictionaries to try
    """
    # For standard MLP, use same network architecture as sequence models for fair comparison
    if not USE_SEQUENCE_ENV or policy_class == "MlpPolicy":
        # Match the architecture used by sequence models
        if model_name.lower() in ['ddpg', 'td3', 'sac']:
            # Off-policy algorithms: use [256, 128] - gradual compression
            return [{"net_arch": [256, 128]}]
        else:
            # On-policy algorithms (PPO, A2C): use default or smaller architecture
            return [{}]
    
    # Extract the feature extractor name from policy class name
    policy_name = policy_class.__name__
    
    # Common policy kwargs that work for all models
    # Increased feature extractor outputs to 256 for better representation capacity
    common_cnn_lstm_kwargs = {
        "features_extractor_kwargs": {
            "cnn_filters": [64, 128],  # Increased from [32, 64]
            "cnn_kernel_sizes": [3, 3],
            "cnn_dropout": 0.1,
            "lstm_hidden_size": 128,  # Increased from 64
            "lstm_num_layers": 1,
            "lstm_dropout": 0.0
        },
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    }
    
    common_cnn_kwargs = {
        "features_extractor_kwargs": {
            "num_filters": [64, 128],  # Increased from [32, 64]
            "kernel_sizes": [3, 3],
            "dropout": 0.1,
            "output_dim": 256  # Increased from 128
        },
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    }
    
    common_lstm_kwargs = {
        "features_extractor_kwargs": {
            "lstm_hidden_size": 256,  # Increased from 128
            "num_layers": 2,
            "dropout": 0.1
        },
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    }
    
    common_transformer_kwargs = {
        "features_extractor_kwargs": {
            "embed_dim": 256,  # Increased from 128
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1
        },
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU
    }
    
    # Model-specific adjustments for off-policy algorithms
    if model_name.lower() in ['ddpg', 'td3', 'sac']:
        # Off-policy algorithms: use [256, 128] for gradual compression
        if 'CNNLSTM' in policy_name:
            return [{**common_cnn_lstm_kwargs, "net_arch": [256, 128]}]
        elif 'CNN' in policy_name:
            return [{**common_cnn_kwargs, "net_arch": [256, 128]}]
        elif 'LSTM' in policy_name:
            return [{**common_lstm_kwargs, "net_arch": [256, 128]}]
        elif 'Transformer' in policy_name:
            return [{**common_transformer_kwargs, "net_arch": [256, 128]}]
    
    # Default for on-policy (PPO, A2C)
    if 'CNNLSTM' in policy_name:
        return [common_cnn_lstm_kwargs]
    elif 'CNN' in policy_name:
        return [common_cnn_kwargs]
    elif 'LSTM' in policy_name:
        return [common_lstm_kwargs]
    elif 'Transformer' in policy_name:
        return [common_transformer_kwargs]
    
    return [{}]


if ENV_TYPE == 'portfolio':
    if USE_SEQUENCE_ENV:
        e_train_gym = StockPortfolioSequenceEnv(df=train, **env_kwargs)
        # Pass training normalization stats to eval env to prevent data leakage
        norm_stats = e_train_gym.get_normalization_stats()
        reward_stats = e_train_gym.get_reward_stats()
        eval_kwargs = dict(**env_kwargs)
        eval_kwargs['random_start'] = False  # Deterministic evaluation
        eval_kwargs['randomize_interval_offset'] = False  # Fixed phase in evaluation
        eval_kwargs['update_reward_stats'] = False  # freeze reward normalization stats during eval
        e_eval_gym = StockPortfolioSequenceEnv(
            df=val,
            normalization_stats=norm_stats,
            reward_stats=reward_stats,
            **eval_kwargs
        )
    else:
        # Use MLP environment (remove sequence-specific kwargs)
        mlp_kwargs = {k: v for k, v in env_kwargs.items() 
                     if k not in ['sequence_length', 'flatten_observations']}
        e_train_gym = StockPortfolioMLPEnv(df=train, **mlp_kwargs)
        # Pass training normalization stats to eval env to prevent data leakage
        norm_stats = e_train_gym.get_normalization_stats()
        reward_stats = e_train_gym.get_reward_stats()
        eval_mlp_kwargs = dict(**mlp_kwargs)
        eval_mlp_kwargs['update_reward_stats'] = False  # freeze reward normalization stats during eval
        eval_mlp_kwargs['random_start'] = False
        eval_mlp_kwargs['randomize_interval_offset'] = False  # Fixed phase in evaluation
        e_eval_gym = StockPortfolioMLPEnv(
            df=val,
            normalization_stats=norm_stats,
            reward_stats=reward_stats,
            **eval_mlp_kwargs
        )
elif ENV_TYPE == 'trading':
    if USE_SEQUENCE_ENV:
        e_train_gym = StockTradingSequenceEnv(df=train, **env_kwargs)
        e_eval_gym = StockTradingSequenceEnv(df=val, **env_kwargs)
    else:
        raise NotImplementedError("MLP environment for trading not yet implemented")

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

env_eval, _ = e_eval_gym.get_sb_env()
print(type(env_eval))

# ...existing code...

# e_train_gym = StockPortfolioSequenceEnv(df = train, **env_kwargs)

# env_train, _ = e_train_gym.get_sb_env()
# print(type(env_train))

# e_eval_gym = StockPortfolioSequenceEnv(df=val,**env_kwargs)

# env_eval, _ = e_eval_gym.get_sb_env()
# print(type(env_eval))

# initialize
agent = DRLAgent(env = env_train)

# Create results subfolder BEFORE hyperparameter search to save/load params
reward_type = env_kwargs.get('reward_type', 'default')
sequence_length = env_kwargs.get('sequence_length', 1)
results_subfolder = create_results_subfolder(
    model_name=model,
    feature_extractor=feature_extractor,
    env_type=ENV_TYPE,
    reward_type=reward_type,
    sequence_length=sequence_length
)

print(f"\nResults will be saved to: {results_subfolder}\n")

# Check if best params already exist
best_params_file = os.path.join(results_subfolder, 'best_hyperparams.json')

if os.path.exists(best_params_file):
    print(f"\n{'='*60}")
    print("⚡ Loading cached hyperparameters (skipping grid search)")
    print(f"{'='*60}\n")
    
    with open(best_params_file, 'r') as f:
        saved_params = json.load(f)
    
    # Extract model_params from saved data
    best_params = saved_params
    
    print("Loaded Best Params: ", best_params)
    
    # Reconstruct policy and policy_kwargs from current configuration
    policy_kwargs = get_policy_kwargs_grid(policy, model)
    best_model = agent.get_model(
        model_name=model,
        policy=policy,
        policy_kwargs=policy_kwargs[0] if policy_kwargs else {},
        model_kwargs=best_params
    )
    
else:
    print(f"\n{'='*60}")
    print("🔍 Running hyperparameter grid search (first time)")
    print(f"{'='*60}\n")
    
    params = get_rl_model_params(model)
    policy_kwargs = get_policy_kwargs_grid(policy, model)  # Pass model name as second argument
    
    # 8) Run search_best_hparams
    best_params, best_model = agent.search_best_hparams(
        model_name=model,
        train_df=train,
        val_df=val,
        param_grid=params,
        policy=policy,
        policy_kwargs_grid=policy_kwargs,
        total_timesteps=TOTAL_TIMESTEPS,
        env_constructor=lambda the_df, **kwargs: make_env(the_df, ENV_TYPE, **kwargs),  # Updated
        eval_freq=5_000,
        best_model_save_path="./best_hparam_search",
        **env_kwargs
    )
    
    print("\nHyperparam Search Results:")
    print("Best Params: ", best_params)
    
    # Extract only serializable model_params for saving
    # best_params contains: {'model_params': {...}, 'policy': <class>, 'policy_kwargs': {...}}
    # We only save model_params since policy/policy_kwargs are reconstructed from configuration
    params_to_save = best_params.get('model_params', best_params)
    
    # Save best params to file for future runs
    with open(best_params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    
    print(f"\n✅ Best hyperparameters saved to: {best_params_file}\n")
    
    # Update best_params to only contain model_params for walk_forward
    best_params = params_to_save

print("We have a best_model trained with these params")

# 9) Once hyperparams are found, proceed with walk-forward:
start_date = "2021-10-01"
end_date   = "2025-12-31"

#   We'll choose a 63-day validation window each iteration, 
#   then a 63-day trading window (rebalance_window).
rebalance_window = 63 
val_window       = 63 

seeds = range(1,21)
for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # Results subfolder already created before hyperparameter search
    print(f"\n{'='*60}")
    print(f"Running walk-forward validation with seed {seed}")
    print(f"Results directory: {results_subfolder}")
    print(f"{'='*60}\n")

    df_res, df_account_all, df_actions_all = agent.walk_forward_final_vs_checkpoint(
        df=df,
        unique_trade_dates=unique_dates,
        start_date=start_date,
        end_date=end_date,
        model_name=model,
        fixed_params=best_params,
        rebalance_window=rebalance_window,
        val_window=val_window,
        total_timesteps=TOTAL_TIMESTEPS,
        env_constructor=lambda the_df, **kwargs: make_env(the_df, ENV_TYPE, **kwargs),  # Updated
        eval_freq=5000,
        best_model_prefix="./walkforward_best_model",
        seed=seed,
        **env_kwargs
    )


    print("Walk-Forward Results Summary:")
    print(df_res)
    df_res.to_csv(os.path.join(results_subfolder, f'walkforward_results_seed{seed}.csv'))
    print("Account Value Memory Over All Trading Windows:")
    print(df_account_all.head())
    df_account_all.to_csv(os.path.join(results_subfolder, f'walkforward_account_value_seed{seed}.csv'))
    print("Actions Memory Over All Trading Windows:")
    print(df_actions_all.head())
    df_actions_all.to_csv(os.path.join(results_subfolder, f'walkforward_actions_seed{seed}.csv'))
    perf_stats_all = backtest_stats(account_value=df_account_all)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv(os.path.join(results_subfolder, f'walkforward_perf_stats_seed{seed}.csv'))


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