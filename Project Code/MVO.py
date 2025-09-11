from pypfopt import EfficientFrontier, risk_models # pip install PyPortfolioOpt
# ! pip install ecos osqp scs cvxopt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import yfinance as yf # conda install conda-forge::yfinance

# maximial sharpe ratio MVO
def calculate_mvo(return_data, risk_free_rate = 0):
    """
    Calculate MVO weights for maximal Sharpe ratio
    """
    trading_days = 252
    mu = return_data.mean(axis=0) * trading_days
    S = np.cov(return_data.T, ddof=0) * trading_days
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    weights = ef.max_sharpe(risk_free_rate = risk_free_rate) # saves to ef object
    cleaned_weights = ef.clean_weights()
    
    return cleaned_weights, mu, S, ef.portfolio_performance(risk_free_rate=risk_free_rate) 

# mvo using black-litterman allocation
def calculate_mvo_bl(return_data, tickers, use_random_market_cap):
    # Absolute views: simulate expected returns
    trading_days = 252
    #absolute_views = {ticker: np.random.normal(0.05, 0.1) for ticker in tickers}
    absolute_views = {ticker: 0.05 for ticker in tickers}
    market_caps = {}
    med_mc_snp500 = 183977 * 1000

    if use_random_market_cap:
        caps_list = np.random.randint(1, 10, size=len(tickers)) # exclusive sample with replacement 
    else:
        caps_list = []
        for ticker in tickers:
            try:
                mc = yf.Ticker(ticker).info.get("marketCap") # assume market cap has not changed much since 2023
                if mc is None: # for 0 value case 
                    mc = med_mc_snp500
            except Exception as e: 
                mc = med_mc_snp500  # for error case 
                print(f"!!!Market-Cap of {ticker} not avaialble so estimated with median S&P 500 market cap!!!")
            caps_list.append(mc)
 
    sum_caps = np.sum(caps_list)
    for i, ticker in enumerate(tickers):
        market_caps[ticker] = caps_list[i] / sum_caps

    epsilon = 1e-1 # pertubate for cov matrix to avoid 0 case
    Sigma = risk_models.sample_cov(return_data) * trading_days
    S_bl = Sigma + epsilon * np.eye(Sigma.shape[0]) # pertubete covariance matrix for numerical stability 
    bl = BlackLittermanModel(S_bl, pi = "market", absolute_views=absolute_views, market_caps=market_caps, risk_aversion=2.5)
    bl_returns = bl.bl_returns()

    ef_bl = EfficientFrontier(bl_returns, S_bl, solver = "OSQP", weight_bounds = (0, 1))
    ef_bl.max_sharpe()
    cleaned_weights_bl = ef_bl.clean_weights()

    return cleaned_weights_bl, caps_list