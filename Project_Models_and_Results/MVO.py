from pypfopt import EfficientFrontier, risk_models # pip install PyPortfolioOpt
# ! pip install ecos osqp scs cvxopt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import numpy as np
import yfinance as yf # conda install conda-forge::yfinance
import requests 
import format_zip_data as fz
from polygon import RESTClient
import streamlit as st 
import matplotlib.pyplot as plt 
import coinbase_trade as cb_trade

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
def calculate_mvo_bl(return_data, tickers, market_cap_method = ["random", "yfinance", "fmp", "polygon"], api_key = None):
    # Absolute views: simulate expected returns
    absolute_views = {ticker: 0.05 for ticker in tickers} # non-bias views
    market_caps = {}
    med_mc_snp500 = 183977 * 1000

    if market_cap_method == "Random":
        caps_list = np.random.randint(1, 10, size=len(tickers)) # exclusive sample with replacement 
    else:
        caps_list = []
        for ticker in tickers:
            mc = 0
            # yfinance not working!
            if market_cap_method == "yfinance" :
                try:
                    mc = yf.Ticker(ticker).info.get("marketCap") # assume market cap has not changed much since 2023
                    if mc is None: # for 0 value case 
                        mc = med_mc_snp500
                except Exception as e: 
                    mc = med_mc_snp500  # for error case 
                    print(f"!!!Market-Cap of {ticker} not avaialble so estimated with median S&P 500 market cap!!!")
            elif market_cap_method == "FMP":
                url = f"https://financialmodelingprep.com/stable/market-capitalization?symbol={ticker}&apikey={api_key}"
                res = requests.get(url)
                mc = res[0]['marketCap']
            elif market_cap_method == "polygon": 
                ticker_method_input = {"ticker": ticker}
                client = RESTClient(api_key = api_key)
                metadata = fz.get_poly(client = client, **ticker_method_input)
                mc = metadata['market_cap'] # get required key 

            caps_list.append(mc)


    sum_caps = np.sum(caps_list)
    for i, ticker in enumerate(tickers):
        market_caps[ticker] = caps_list[i] / sum_caps

    epsilon = 1e-1 # pertubate for cov matrix to avoid 0 case
    Sigma = risk_models.risk_matrix(prices = return_data, returns_data = True, method = 'sample_cov')
    S_bl = Sigma + epsilon * np.eye(Sigma.shape[0]) # pertubete covariance matrix for numerical stability 
    bl = BlackLittermanModel(S_bl, pi = "market", absolute_views=absolute_views, market_caps=market_caps, risk_aversion=2.5)
    bl_returns = bl.bl_returns()

    ef_bl = EfficientFrontier(bl_returns, S_bl, solver = "OSQP", weight_bounds = (0, 1))
    ef_bl.max_sharpe()
    cleaned_weights_bl = ef_bl.clean_weights()

    return cleaned_weights_bl, caps_list

# Will simulate many portfolios and color by sharpe ratio (ordinal scale), and trace out efficient frontier
def mvo_max_sharpe_plot(mean_returns, covariance_matrix, tickers):
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.title("Mean-Variance Optimisation (MVO) Problem")

    ef = EfficientFrontier(mean_returns, covariance_matrix, weight_bounds=(0, 1))
    ef_max_sharpe = ef.deepcopy()
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # optimal portfolo
    ef_max_sharpe.max_sharpe()
    return_tangent, volatility_tangent, _ = ef_max_sharpe.portfolio_performance(risk_free_rate = 0)
    ax.scatter(volatility_tangent, return_tangent, marker="*", s=100, c="r", label="Max Sharpe/Tangency Portfolio")

    # number of random portfolios 
    n_samples = st.number_input("**Number of Random Portfolios to Generate**", min_value = 1, max_value = 10000, value = 3000)

    # generate random portfolios
    w = np.random.dirichlet(np.ones(len(tickers)), n_samples) # random portfolio weights 
    returns = w.dot(mean_returns)
    volatilities = np.sqrt(np.diag(w @ covariance_matrix @ w.T))
    sharpes = returns / volatilities

    ax.scatter(volatilities, returns, c=sharpes, cmap="inferno")
    fig.tight_layout()
    ax.legend()

    return fig 