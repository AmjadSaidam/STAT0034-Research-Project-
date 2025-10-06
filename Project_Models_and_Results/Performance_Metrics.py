import pandas as pd
import numpy as np 
import DLS_model as dls

# =============================================
#  Porfolio Out of Sample Returns: Continuous Compounding 
# =============================================

def get_returns(weights, returns, window, target_volatility, cost = 0.0001, use_vol_scaling = True):
    rolling_volatility = dls.std_ema(returns, window)
    volatility_scale = (target_volatility / rolling_volatility)
    transaction_cost = 0
    if use_vol_scaling:
        if cost > 0:
            transaction_cost = np.abs(rolling_volatility*weights - np.roll(rolling_volatility, 1)*np.roll(weights, 1)).sum(axis = 1)
        else:
            pass
        
        adjsuted_returns = (weights * returns * volatility_scale).sum(axis = 1) - cost * transaction_cost
    else:
        adjsuted_returns = (weights * returns).sum(axis = 1)

    return adjsuted_returns

# =============================================
# Equity Returns
# =============================================

def equity_curve(returns):
    """Calculate equity curve from returns series (equation 26)"""
    if len(returns) == 0:
        return pd.Series([1.0])
    eqty = (1 + returns).cumprod() - 1
    return eqty

# =============================================
# Sharpe Ratios
# =============================================

def calculate_sharpe_ratio(returns, annualization_factor=252**0.5, risk_free_rate=0):
    """Calculate annualized Sharpe ratio."""
    return ((np.mean(returns) - risk_free_rate) / np.std(returns)) * annualization_factor

# =============================================
# Sortino Ratio
# =============================================

def calculate_sortino_ratio(returns, annualisation_factor = 252**0.5, risk_free_rate = 0): 
    returns_mean = np.array(returns).mean()
    downside_adjsuted_returns = np.minimum(0, np.array(returns) - risk_free_rate/252)
    downside_deviation = np.sqrt((1/(len(returns)-1))*np.sum((downside_adjsuted_returns)**2)) 
    return (returns_mean / downside_deviation) * annualisation_factor

# =============================================
# VaR and CVaR
# =============================================

# value at risk 
def VaR(returns, alpha = 0.05): 
    if not isinstance(returns, pd.Series):
        returns.flatten() 
    return np.quantile(returns, alpha)

# conditional value at risk 
def CVaR(returns, alpha = 0.05):
    var = VaR(returns, alpha)
    return returns[returns <= var].mean()