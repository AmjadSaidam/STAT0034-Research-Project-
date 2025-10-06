# =============================================
# Imports 
# =============================================
# To verify if module installed call <pip show {package_name}>
# general 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import format_zip_data as fz 
# import models
import torch
import MVO as mvo
import DmNN as dmnn 
import DmLSTM as dmlstm 
import DLS_model as dls 
import SaveLoadTorchModel as sl_torch # load / save models
import coinbase_trade as cb_trade
import Performance_Metrics as pm 
from scipy.stats import skewnorm
# others
import datetime  
import json
# Fix imports 

# =============================================
# Global 
# =============================================

# Streamlit app configuration
st.set_page_config(page_title="Portfolio Optimization App", layout="wide")

# Title
st.title("Portfolio Optimisation using Distribution-Based Deep Learning Models")

# =============================================
# Sidebar, user inputs  
# =============================================

def keys_to_session_state(keys, type = [False, None]):
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = type

def init_session_state():
    defaults = {
        'fmp_polygon_data': False,
        'cb_data_button': False,
        'pulled_fmp_polygon_data': False,
        'pulled_cb_data': False,
        'return_data': None,
        'in_sample_returns': None,
        'out_of_sample_returns': None,
        'price_data': None,
        'in_sample_prices': None,
        'out_of_sample_prices': None,
        'tickers': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def save_data(return_data, in_sample_returns, out_of_sample_returns,
              price_data, in_sample_prices, out_of_sample_prices, tickers):
    st.session_state.return_data = return_data
    st.session_state.in_sample_returns = in_sample_returns
    st.session_state.out_of_sample_returns = out_of_sample_returns
    st.session_state.price_data = price_data
    st.session_state.in_sample_prices = in_sample_prices
    st.session_state.out_of_sample_prices = out_of_sample_prices
    st.session_state.tickers = tickers

def plot_model_trained_weights(model_weights, title):
    model_weights = model_weights.flatten()
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.title(title)
    plt.hist(model_weights, bins = 50, density = True)
    plt.xlabel("Weight Values")
    plt.ylabel("Frequency Density")
    plt.tight_layout()
    st.pyplot(fig, clear_figure = False)   

# set defualt to long_only 
def portfolio_type_button(box_key: str, model_constraint):
    st.selectbox("**Portfolio Constraint**", ["long_only", "long_short"], key=box_key)
    if st.session_state[box_key] == 'long_only':
        model_constraint = False # by default 
    else:
        model_constraint = True

    return model_constraint  

init_session_state()

# Porfolio Settings 
# =============================================
st.sidebar.title(":blue[Portfolio Settings]")

# Initialize session state for the toggle
if "custom_portfolio" not in st.session_state:
    st.session_state.custom_portfolio = False

# Sidebar button to flip the toggle
if st.sidebar.button("Use Custom Portfolio"):
    st.session_state.custom_portfolio = not st.session_state.custom_portfolio
custom_portfolio = st.session_state.custom_portfolio # change to simpler name 

start_date = None 
end_date = None 
user_tickers = []
if custom_portfolio: 
    custom_tick_list = "ETH-GBP, SOL-GBP, LINK-GBP"
    user_tickers = st.sidebar.text_input("Define your Ticker list below (comma seperated)", value = custom_tick_list)
    # user defined tickers to list 
    tickers = [t.strip() for t in user_tickers.split(",")]

    # FMP / Polygon 
    # =============================================
    start_date = st.sidebar.date_input("Start Date", value = datetime.date(2025, 1, 1))
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = st.sidebar.date_input("End Date", value = datetime.date.today())
    end_date = end_date.strftime("%Y-%m-%d")
    user_api_key = st.sidebar.text_input("Client API key (Non Exchange)", value = "YOUR_API_KEY_HERE")
    if st.sidebar.button("Pull Data from FMP/Polygon API"):
        st.session_state.fmp_polygon_data = True
        st.session_state.cb_data_button = False  # disable the other option    

    # Coinbase 
    upload_file = st.sidebar.file_uploader("CDP API key")
    if upload_file is not None:
        data = json.load(upload_file)
        api_name = data['name']
        api_secret = data['privateKey']

    # Attempt Account Connection
    if 'attempt_link' not in st.session_state:
        st.session_state.attempt_link = False

    # connect to account 
    if st.sidebar.button("Login to Coinbase Advanced"): 
        st.session_state.attempt_link = True

    if st.sidebar.button("Pull Coinbase Data"):
        st.session_state.cb_data_button = True
        st.session_state.fmp_polygon_data = False 

    coinbase_acc = None 
    if st.session_state.attempt_link: 
        coinbase_acc = cb_trade.CoinbaseTrader(api_key = api_name, api_secret = api_secret)
        # instentiate class
        st.session_state.cb_login = True
        st.markdown("## Coinbase Account Link and Data Request")
        st.write("Account Link Attempted:", st.session_state.cb_login)

        login_status = coinbase_acc.login()
        if login_status:
            st.success('Successfully connected to Coinbase Advanced Account')
        else:
            st.write("Login Status:", login_status)

        # have we attempted link 
        st.session_state.attempt_link = True
        
else:
    user_api_key = ""

# General Settings
# =============================================
st.sidebar.header('General Settings')

volatility_scaling_window = st.sidebar.number_input("Window for Volatility Scaling", value = 50) # implement in all models 

# Trading 
# =============================================
st.sidebar.header("Trading via Coinbase")

if 'crypto_tickers' not in st.session_state:
    st.session_state.crypto_tickers = "ETH-GBP, SOL-GBP, LINK-GBP" # This is used to send orders 
crypto_ticker_sidebar = st.sidebar.text_input("**Coinbase Cryptocurrency Assets**", 
                                            value = st.session_state.crypto_tickers, 
                                            key = 'crypto_ticker_text_input')
st.session_state.crypto_tickers = st.session_state.crypto_ticker_text_input

# Appendix 
# =============================================
st.sidebar.header("Appendix")

st.sidebar.markdown(
    r"""
    ### Live Trading 📈
    [Coinbase Advanced](https://www.coinbase.com/advanced-trade/spot/BTC-USD) $\\$
    [Trading Doc - READ ME](https://drive.google.com/drive/folders/12fSaZxkolMc1OjZWE0VBldDxrb66JD8O?usp=drive_link)

    ### Data resources 📊
    [FMP Stock API](https://site.financialmodelingprep.com/developer/docs/dashboard) $\\$
    [polygon.io](https://polygon.io/) $\\$

    ### Socials 🌍 $\\$
    [LinkedIn](https://www.linkedin.com/in/amjadsaidam/) $\\$
    [Medium](https://medium.com/@amjadsaidama) $\\$
    [TradingView](https://www.tradingview.com/u/Amjad_S/#published-scripts) $\\$
    """
)

# =============================================
# Get Data 
# =============================================

if not custom_portfolio:
    try:
        # get study data. Note no need to save varaibles as re-runs will yields same data
        data_path = r'Data/Most Watched Stocks.zip' # relative data path 
        return_data, in_sample_returns, out_of_sample_returns = fz.process_zip_with_csvs(data_path, 
                                                                                column_extract = 'Close')
        price_data, in_sample_prices, out_of_sample_prices = fz.process_zip_with_csvs(data_path, 
                                                                                column_extract = 'Close',
                                                                                get_prices = True) 
        save_data(return_data, in_sample_returns, out_of_sample_returns, 
                  price_data, in_sample_prices, out_of_sample_prices, return_data.columns.tolist()) # tickers are not inputed by user so get columns 

        # reset so we can toggle the custom portfolio button 
        st.session_state.pulled_cb_data = False
        st.session_state.pulled_fmp_polygon_data = False

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else: 
    if st.session_state.fmp_polygon_data and not st.session_state.pulled_fmp_polygon_data:
        return_data, in_sample_returns, out_of_sample_returns = fz.polygon_data(
            poly_rest_api_key=user_api_key, tickers=tickers, 
            start_date=start_date, end_date=end_date
        )
        price_data, in_sample_prices, out_of_sample_prices = fz.polygon_data(
            poly_rest_api_key=user_api_key, tickers=tickers, 
            start_date=start_date, end_date=end_date, get_returns=False
        )
        save_data(return_data, in_sample_returns, out_of_sample_returns,
                price_data, in_sample_prices, out_of_sample_prices,
                tickers)
        st.session_state.pulled_fmp_polygon_data = True

    elif st.session_state.cb_data_button and not st.session_state.pulled_cb_data:
        return_data, in_sample_returns, out_of_sample_returns = coinbase_acc.coinbase_data(
            products=tickers, time_frame_candle='ONE_DAY', get_returns=True, days=350
        )
        price_data, in_sample_prices, out_of_sample_prices = coinbase_acc.coinbase_data(
            products=tickers, time_frame_candle='ONE_DAY', get_returns=False, days=350
        )
        save_data(return_data, in_sample_returns, out_of_sample_returns,
                price_data, in_sample_prices, out_of_sample_prices,
                tickers)
        st.session_state.pulled_cb_data = True

# Use saved data if available ---
if st.session_state.return_data is not None:
    return_data = st.session_state.return_data
    in_sample_returns = st.session_state.in_sample_returns
    out_of_sample_returns = st.session_state.out_of_sample_returns
    price_data = st.session_state.price_data
    in_sample_prices = st.session_state.in_sample_prices
    out_of_sample_prices = st.session_state.out_of_sample_prices    
    tickers = st.session_state.tickers

# DL model (DmNN, DmLSTM) prediction data 
# =============================================
out_of_sample_lagged_returns = np.roll(out_of_sample_returns, 1, axis = 0) # lag returns to avoid lookahead bias 

# Data EDA
# =============================================
st.markdown("## Data")

with st.expander('Return Data'):
    st.text("Asset return dataframe") 
    st.dataframe(return_data) 

with st.expander("Price data"):
    st.text("Asset price dataframe")
    st.dataframe(price_data)     

# Equity curves
# =============================================
st.selectbox("**Select Ticker from Data for EDA**", tickers, key = 'selected_ticker')

keys_to_session_state(keys = ['chosen_asset_return', 'chosen_asset_price', 'chosen_asset_equity'], type = None)

if 'printed_asset_eda' not in st.session_state:
    st.session_state.printed_asset_eda = False

if st.button("Print Selected Ticker Data"):
    st.session_state.printed_asset_eda = True

if st.session_state.printed_asset_eda:
    st.session_state.chosen_asset_return = return_data.loc[:, st.session_state.selected_ticker]
    st.session_state.chosen_asset_price = price_data.loc[:, st.session_state.selected_ticker]

# to normnal 
chosen_asset_return = st.session_state.chosen_asset_return
chosen_asset_price = st.session_state.chosen_asset_price

if chosen_asset_return is not None:
    st.session_state.chosen_asset_equity = pm.equity_curve(st.session_state.chosen_asset_return) 
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 5))
    fig.suptitle(f"{st.session_state.selected_ticker}: Simple Returns EDA")

    ax[0, 0].plot(chosen_asset_return)
    ax[0, 0].set_xlabel("Time (days)")
    ax[0, 0].set_ylabel('Simple Returns')

    ax[0, 1].hist(chosen_asset_return, bins = 50)
    ax[0, 1].set_xlabel("Simple Returns")
    ax[0, 1].set_ylabel('Frequency Denisty')
    
    ax[1, 0].plot(st.session_state.chosen_asset_equity)
    ax[1, 0].set_xlabel("Time (days)")
    ax[1, 0].set_ylabel("Cumulative Simple Return")

    ax[1, 1].plot(chosen_asset_price) # add price plot
    ax[1, 1].set_xlabel("Time (days)")
    ax[1, 1].set_ylabel('Close Price')

    plt.tight_layout()
    st.pyplot(fig, clear_figure = False)

# =============================================
# MVO model
# =============================================
st.markdown("## Mean-Variance Optimisation (MVO)")

st.markdown(
    r"""
    The mean-varaince problem outlined by Markowitz (1952) defines the portfolion optimisation problem as follows:

    $$
    \min_{\mathbf{w}} \left[ \mathbf{w}^\top \Sigma \mathbf{w} \right] 
    = \min_{\mathbf{w}} \left[ \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij} \right] \quad 
    \text{subject to} \quad \mathbf{w}^\top \boldsymbol{\mu} = \mu_p; \quad
    \sum_{i=1}^n w_i = 1
    $$

    Where $\mu_p$ is the desired expected portfolio return and $0 \leq w_i \leq 1$, we can solve the above problem to get the optimal portfolio weights for some desired return $\mu_p$, $\hat{\textbf{w}} \in \mathbb{R}^n$.
    """
)

# MVO Efficient Frontier Plot 
# =============================================
mvo_weights_dict, mu, S, _ = mvo.calculate_mvo(return_data = in_sample_returns, risk_free_rate = 0)

# plot fancy efficient frontier and radnom portfolios 
fig = mvo.mvo_max_sharpe_plot(mean_returns=mu, covariance_matrix=S, tickers=tickers)
st.pyplot(fig)

# =============================================
# MVO+BL 
# =============================================
st.markdown("## Mean Variance Optimisation using Black-Litterman Expected Portfolio Returns (MVO+BL)")

st.markdown(
    r"""
    The mean variance problem, using black litterman allocation of expected returns remains unchanged from the 
    original Markowitz (1592) formulation, however we redefine the expected returns vector using a Bayesian approach.  

    $$
    \mu_{BL} 
    = \mathbb{E}[\mathbf{R}] 
    = \left[ \left( \tau \sum \right)^{-1} + P^\top \Omega^{-1} P \right]^{-1} 
    \left[ \left( \tau \sum \right)^{-1} \Pi + P^\top \Omega^{-1} Q \right]
    $$

    Where: $\\$
    $\tau$ = Is a scaler tuning constant $\\$
    $P$ = Asset view matrix $\\$
    $\Omega$ = Diagonal covariance matrix of error terms $\\$
    $\Pi$ = Implied equilibrium return vector $\\$
    $Q$ = View returns vector $\\$ 

    """
    )

with st.expander("Black-Litterman Allocation Derivation"):
    st.markdown(
        r"""
        **Black-Litterman Model and Bayesian Updating:**

        Noting bayes Therom for probability densities

        $$
        p(\theta | y) = \frac{p(\theta \cap y)}{p(y)} = \frac{p(y | \theta) p(\theta)}{\int p(y | \theta) p(\theta) d \theta}
        $$

        **Prior**

        We can model the uncertainty around the implied return vector, $\Pi$, using a multivariate normal distribution. 
        The uncertainity is encoded in the random varaible $r$, the vector of expected excess returns. 

        $$
        r \sim N \left( \mu = \Pi, \sigma^2 = \tau \sum \right)
        $$

        Where: $\\$
        $\Pi$ = The mean of our distribution, which is the implied equailibrium return vector $\\$
        $\tau \sum$ = The scaled covariance matrix of exces returns

        **Likelihood**

        This is the views vector, and tells us the probability of observing the actual data, given our prior beliefs. 

        $$
        Q = Pr + \epsilon, \epsilon \sim N(0, \Omega)
        $$

        **Posterior**


        Applying Bayes theorem, our updated denisty is
        
        $$
        p(r | Q) \propto p(Q | r) p(r)
        $$

        Identifying the kernel we find the functional form of our posterior distribution is 

        $$
        r | Q \sim N \left( \mu = \mu_{BL}, \sigma^2 = \sum_{BL} \right)
        $$

        So the normal distribution of $r$ is a conjugate prior for the likelihood $Q | r$, meaning our posterioir distribution 
        is also normally distributed (same functional family as prior).  
        
        **Optimisation Problem**

        The updated mean-variance optimisation problem. The optimal weights can be found be substiuting the new expected return vector 
        in the solution of the original MVO problem.

        $$
        \min_{\mathbf{w}} \left[ \mathbf{w}^\top \Sigma \mathbf{w} \right] \quad
        \text{subject to} \quad \mathbf{w}^\top \boldsymbol{\mu} = \mu_{BL}; \quad
        \sum_{i=1}^n w_i = 1
        $$
        """
    )

# =============================================
# Target Distribution
# =============================================

st.markdown("## Target Distribution")
target_mean = st.slider("**Target Distribution Scale (Mean)**", value = 0.01)
target_std = st.slider("**Target Distribution Location (Standard Deviation)**", value = 0.04)
target_skewness = st.slider("**Target Distribution Skewness**", value = 3, min_value = -10, max_value = 10) # for normal set to 0 

# insure target dist is constant oon re-runs
np.random.seed(1)
target_returns = skewnorm.rvs(a = target_skewness, loc = target_mean, scale = target_std, size=(len(in_sample_returns), 1))

# plot figure
fig, ax = plt.subplots(figsize = (10, 5))
plt.title("Target Return Density")
plt.hist(target_returns, bins = 50, density=True)
plt.xlabel('Sampled Returns')
plt.ylabel("Frequency Denisty")
plt.tight_layout()
st.pyplot(fig)

# =============================================
# DmNN 
# =============================================
st.markdown("## Distance-metric Neural Network (DmNN)")

# Initialize session state variables
# =============================================
keys_to_session_state(keys = ['dmnn_vol_scaling', 'dmnn_portfolio_type', 'dmnn_training_complete', 'dmnn_run'], type = False)
keys_to_session_state(keys = ['dmnn_trained_weights', 'dmnn_trained_model', 'dmnn_predicted_weights', 'dmnn_out_sample_returns'], type = None)

# UI's
# =============================================
loss_function_dmnn = st.selectbox("**Loss Function**", ["l2", "mmd", "wasserstein"], key = 'dmnn_loss')
dmnn_hidden_layers = st.slider("**Hidden Layers**", min_value = 1, value = 2, max_value = 10, key = 'dmnn_hidden_layers')
dmnn_number_epochs = st.number_input("**Number of Epochs**", value = 1000, min_value=1, max_value=10000)

if st.button("Use Volatility Scaling", key = 'dmnn_vol_scaling_btn'):
    st.session_state.dmnn_vol_scaling = not st.session_state.dmnn_vol_scaling
st.write("Using volatility scaling for DmNN?", st.session_state.dmnn_vol_scaling)

dmnn_pf_type = portfolio_type_button(box_key = 'dmnn_portfolio_type_box', model_constraint=st.session_state.dmnn_portfolio_type)

# Model Training
# =============================================
if st.button("Run Model", key = 'run_dmnn'):
    st.session_state.dmnn_run = True
    st.session_state.dmnn_training_complete = False
st.write("Training Status:", st.session_state.dmnn_run)

# abstract this proess away, pass the model to be trained as model with kwargs and dictionary output
if st.session_state.dmnn_run and not st.session_state.dmnn_training_complete:
    with st.spinner("Training DmNN model"):
        _, _, dmnn_trained_weights, _, _, dmnn_trained = dmnn.backprop(in_sample_returns, 
                                                                        target_returns, 
                                                                        target_volatility = target_std, 
                                                                        hidden_layers = dmnn_hidden_layers, 
                                                                        neurons_per_layer = 100, 
                                                                        use_vol_scaling = st.session_state.dmnn_vol_scaling, 
                                                                        long_short = dmnn_pf_type, 
                                                                        distance_metric = loss_function_dmnn, 
                                                                        n_epochs = dmnn_number_epochs, 
                                                                        show_progress = True)
        st.session_state.dmnn_trained_weights = dmnn_trained_weights
        st.session_state.dmnn_trained_model = dmnn_trained
        st.session_state.dmnn_training_complete = True
                
        # Prediction
        if st.session_state.dmnn_trained_model is not None:
            st.session_state.dmnn_predicted_weights = st.session_state.dmnn_trained_model(
                torch.FloatTensor(out_of_sample_lagged_returns))

if st.session_state.dmnn_training_complete:      
    st.success("Training of DmNN model complete")

# Use session state variables for consistent access
dmnn_trained_weights = st.session_state.dmnn_trained_weights
dmnn_trained_model = st.session_state.dmnn_trained_model
predicted_weights_dmnn = st.session_state.dmnn_predicted_weights

# Reset button
if st.session_state.dmnn_trained_weights is not None:
    if st.button("Reset DmNN Training"):
        st.session_state.dmnn_trained_weights = None
        st.session_state.dmnn_trained_model = None
        st.session_state.dmnn_predicted_weights = None
        st.session_state.dmnn_training_complete = False
        st.session_state.dmnn_run = False
        st.rerun() # hault and re-run script 

# DmNN EDA
# =============================================
st.write("### DmNN Trained Model EDA") 

if dmnn_trained_weights is not None:
    plot_model_trained_weights(dmnn_trained_weights, title = 'Histogram of DmNN Trained Model Weights')
    st.session_state.dmnn_out_sample_returns = pm.get_returns(weights = predicted_weights_dmnn.detach().numpy(), 
                                    returns = out_of_sample_returns, 
                                    window = 50, 
                                    target_volatility = target_std, 
                                    use_vol_scaling = st.session_state.dmnn_vol_scaling)
    
    dmnn_sl = sl_torch.SaveLoadPyTorchModel(model = dmnn_trained_model)
    dmnn_buffer = dmnn_sl.save_model()
    st.download_button(
        'Download Entire Trained DmNN Model',
        data = dmnn_buffer, 
        file_name = 'dmnn_trained_model.pth', 
        mime="application/octet-stream"
    )
else:
    st.info("DmNN not Trained yet. Click the 'Run Model' button to train, view model EDA and Download Model")

dmnn_out_sample_returns = st.session_state.dmnn_out_sample_returns

# =============================================
# DmLSTM 
# =============================================
st.markdown("## Distance-metric Long-Short Term Memory (DmLSTM)")

keys_to_session_state(keys = ['dmlstm_vol_scaling', 'dmlstm_portfolio_type', 'dmlstm_training_complete', 'run_dmlstm'], type = False)
keys_to_session_state(keys = ['dmlstm_trained_weights', 'dmlstm_trained_model', 'dmlstm_predicted_weights', 'dmlstm_out_sample_returns'], type = None)

# UI Controls
# =============================================
loss_function_dmlstm = st.selectbox("**Loss Function**", ["l2", "mmd", "wasserstein"], key='dmlstm_loss')
dmlstm_hidden_dim = st.number_input("**Hidden Dimensions**", min_value=1, value=100, max_value=1000)
dmlstm_hidden_layers = st.slider("**Hidden Layers**", min_value=1, value=2, max_value=10, key='dmlstm_hidden_layers')
dmlstm_number_epochs = st.number_input("**Number of Epochs**", value=100, min_value=1, max_value=1000)

if st.button("Use Volatility Scaling", key='dmlstm_vol_scaling_btn'):
    st.session_state.dmlstm_vol_scaling = not st.session_state.dmlstm_vol_scaling
st.write("Using volatility scaling for DmLSTM?", st.session_state.dmlstm_vol_scaling)

dmsltm_pf_type = portfolio_type_button(box_key='dmlstm_portfolio_type_box', model_constraint = st.session_state.dmlstm_portfolio_type)

# Model Training
# =============================================
if st.button("Run Model", key='dmlstm_run_toggle'):
    st.session_state.run_dmlstm = True
    st.session_state.dmlstm_training_complete = False  # Reset completion flag
st.write("Training Status:", st.session_state.run_dmlstm)

# Train only when button pressed and training not completed
if st.session_state.run_dmlstm and not st.session_state.dmlstm_training_complete:
    with st.spinner("Training DmLSTM model..."):
        _, _, dmlstm_trained_weights, _, _, dmlstm_trained_model = dmlstm.backprop_rnn(
            asset_returns=in_sample_returns, 
            target_returns=target_returns, 
            target_volatility=target_std, 
            hidden_dim=dmlstm_hidden_dim,
            num_layers=dmlstm_hidden_layers, 
            use_vol_scaling=st.session_state.dmlstm_vol_scaling, 
            long_short=dmsltm_pf_type, 
            distance_metric=loss_function_dmlstm,
            n_epochs=dmlstm_number_epochs, 
            show_progress=True
        )

        # Save to session_state
        st.session_state.dmlstm_trained_weights = dmlstm_trained_weights
        st.session_state.dmlstm_trained_model = dmlstm_trained_model
        st.session_state.dmlstm_training_complete = True
        
        # Generate predictions
        if st.session_state.dmlstm_trained_model is not None:
            st.session_state.dmlstm_predicted_weights = st.session_state.dmlstm_trained_model(
                torch.FloatTensor(out_of_sample_lagged_returns)
            )

if st.session_state.dmlstm_training_complete: 
    st.success("DmLSTM model training completed!")

# Use session state variables for consistent access
dmlstm_trained_weights = st.session_state.dmlstm_trained_weights
dmlstm_trained_model = st.session_state.dmlstm_trained_model
predicted_weights_dmlstm = st.session_state.dmlstm_predicted_weights

# Reset button
if st.session_state.dmlstm_trained_weights is not None:
    if st.button("Reset DmLSTM Training"):
        st.session_state.dmlstm_trained_weights = None
        st.session_state.dmlstm_trained_model = None
        st.session_state.dmlstm_predicted_weights = None
        st.session_state.dmlstm_training_complete = False
        st.session_state.run_dmlstm = False
        st.rerun()

# DmLSTM EDA - Always show if we have trained weights
# =============================================
st.write("### DmLSTM Trained Model EDA")

if dmlstm_trained_weights is not None:
    plot_model_trained_weights(model_weights=dmlstm_trained_weights, title='Histogram of DmLSTM Trained Model Weights')

    # out of sample returns 
    st.session_state.dmlstm_out_sample_returns = pm.get_returns(weights = predicted_weights_dmlstm.detach().numpy(), 
                                    returns = out_of_sample_returns, 
                                    window = 50, 
                                    target_volatility = target_std, 
                                    use_vol_scaling = st.session_state.dmnn_vol_scaling)

    # save custom model locally 
    dmlstm_sl = sl_torch.SaveLoadPyTorchModel(model = dmlstm_trained_model)
    dmlstm_buffer = dmlstm_sl.save_model()
    st.download_button(
        'Download Entire Trained DmLSTM Model',
        data = dmlstm_buffer, 
        file_name = 'dmlstm_trained_model.pth', 
        mime="application/octet-stream"
    )
else:
    st.info("No DmLSTM model trained yet. Click 'Run Model' to train, view model EDA and Download Model.")

dmlstm_out_sample_returns = st.session_state.dmlstm_out_sample_returns

# =============================================
# DLS  
# =============================================
st.markdown("## DLS [10]")

keys_to_session_state(keys = ['dls_vol_scale', 'train_dls', 'dls_training_complete'], type = False) # long only model with no portfolio type
keys_to_session_state(keys = ['dls_weights', 'dls_trained_model', 'dls_out_sample_features', 'dls_predicted_weights', 'dls_out_sample_returns'], type = None)

n_epochs_dls = st.number_input("**Number of Epochs for DLS**", min_value=1, value=100, max_value=1000)

# volatility scaling
if st.button("Use Volatility Scaling", key = 'dls_vol_scale_button'):
    st.session_state.dls_vol_scale = not st.session_state.dls_vol_scale
st.write("Using volatility scaling for DLS:", st.session_state.dls_vol_scale)

# Train DLS
# =============================================
if st.button("Run Model", key='dls_train_toggle'):
    st.session_state.train_dls = True
    st.session_state.dls_training_complete = False  # Reset completion flag when starting new training

st.write("Training Status:", st.session_state.train_dls)

# Training logic - only run if we need to train and haven't completed training yet
if st.session_state.train_dls and not st.session_state.dls_training_complete:
    with st.spinner("Training DLS model..."):
        dls_trained_weights, trained_dls_model = dls.train_model(
            asset_returns=pd.DataFrame(in_sample_returns).values, 
            asset_prices=pd.DataFrame(in_sample_prices).values, 
            n_epochs=n_epochs_dls, 
            target_volatility=target_std, 
            vol_scaling=st.session_state.dls_vol_scale, 
            show_progress=True
        )
        
        # Store results in session state
        st.session_state.dls_weights = dls_trained_weights
        st.session_state.dls_trained_model = trained_dls_model
        st.session_state.dls_training_complete = True  # Mark training as complete

        # get predictions
        if st.session_state.dls_trained_model is not None: 
            st.session_state.dls_out_sample_features, _ = dls.prepare_features(asset_returns = out_of_sample_returns, 
                                            asset_prices = out_of_sample_prices, 
                                            lookback = 50)
            st.session_state.dls_predicted_weights = st.session_state.dls_trained_model(
                st.session_state.dls_out_sample_features
            )

if st.session_state.dls_training_complete is not None:
    st.success("DLS model training completed!") # print if training ran without errors

# Use session state variables for consistent outputs
dls_trained_weights = st.session_state.dls_weights
dls_predicted_weights = st.session_state.dls_predicted_weights

# Optional: Add reset button
if st.session_state.dls_weights is not None:
    if st.button("Reset Training"):
        st.session_state.train_dls = False
        st.session_state.dls_training_complete = False
        st.session_state.dls_weights = None
        st.session_state.dls_trained_model = None
        st.session_state.dls_out_sample_features = None
        st.rerun()

# DLS EDA - Only show if we have trained weights
# =============================================
st.write("### DLS Trained Model EDA")

if dls_trained_weights is not None:
    plot_model_trained_weights(model_weights=dls_trained_weights, title = 'Histogram of DLS Trained Model Weights')
    
    # get out of sample returns
    st.session_state.dls_out_sample_returns = pm.get_returns(weights = dls_predicted_weights.detach().numpy(), 
                                    returns = out_of_sample_returns[50:], # because of batching firsdt lookback number of bars are unavailable (in paper we do not use this method) 
                                    window = 50, 
                                    target_volatility = target_std, 
                                    use_vol_scaling = st.session_state.dls_vol_scale)

else:
    st.info("No DLS model trained yet. Click 'Run Model' to train the model and see the EDA.")

dls_out_sample_returns = st.session_state.dls_out_sample_returns

# =============================================
# Coinbase Automatig Trading   
# =============================================
st.markdown("## Trading Models:")

st.markdown(
    r"""

    This chapter allows us to implement either of the novel Deep Learning model's saved above on real data, with real money, 
    exclusively for cryptocurrency markets accessed through the Coinbase Exchange. You will require a coinbase advanced
    account with a non-zero account balance to place market orders. 
    
    *Before using any of the features in this section it is highly recommended you read the trading doc under the "Live Trading"
    heading in the Appendix*. 

    """
)

if st.session_state.attempt_link:
    # get some account data
    st.write("**User Live Accounts**\n", coinbase_acc.get_user_accounts())
    
    st.markdown(
        r"""
        The dictionary above displays your coinbase accout information. Once we have authenicated login we use the $\verb|get_accounts()|$ 
        coinbase API endpoint to get the base value invested in all assets in our account, i.e. the assets you invested in and the currency 
        amount that you invested. Note if the dictionary shows a zero value, except for you base account (the account with the currency 
        which you depost funds to your account in) then you have not yet created a portfolio or have closed all filled and open orders. 

        It also important to note that only the assets you define in the "Coinbase Cryptocurrency Assets" under the 
        "Trading via Coinbase" heading can be edited directly from this page. 

        ### Trading Restrictions

        - No volatility scaling enabeld models 
        - No long-short constraint specified models
        """
    )

# Get DL Model Data 
# =============================================
st.markdown("### Get Deep Learning Model Data")

st.write("Press this button to pull data before loading in you DL model.")

if 'change_price_data' not in st.session_state:
    st.session_state.change_price_data = None
if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = None

# Button to trigger data pull
if st.button("Pull Percentage Change and Price Data"):
    crypto_tickers_list = [t.strip() for t in st.session_state.crypto_tickers.split(',')]
    st.session_state.change_price_data = coinbase_acc.get_asset_changes_price(crypto_tickers_list)
    st.session_state.portfolio_tickers = crypto_tickers_list

# Always display data if it exists in session state
if st.session_state.change_price_data is not None:
    change = st.session_state.change_price_data['change']
    price = st.session_state.change_price_data['price']

    st.markdown("**Daily Asset Percentage Change Data**")
    with st.expander("Daily Asset Percentage Change Data"):
        st.markdown(
            """
            The cryptocurrency 24h change in price calculauted using simple returns. Both our novel DL solutions
            take this a feature input exclusivley (DLS also takes this as input, along with price). 
            """
        )
        st.dataframe(change)
    with st.expander("Daily Asset Price Data"):
        st.markdown(
            """
            Asset associated end of day price data. Only DLS makes use of this data directly (simple returns are 
            derived from this and the price data directley prior)
            """
        )
        st.dataframe(price)

# Load Saved Models
# =============================================
st.markdown("### Load saved Deep Learning Models")

# Upload desired deep learning model
dl_upload = st.file_uploader("**Load Deep Learning (DL) Model**", type = ['pth', 'pt'], key = 'upload_dl_model') # pth = pickle

if dl_upload is not None:
    dl_model = torch.load(dl_upload, weights_only = False)

    st.success("Model Was Loaded Succesfully and is Ready for Trading")
else:
    st.info("Upload Saved Model to Generate Market Order")

change_input = torch.FloatTensor(change)

st.write("**DmNN and DmLSTM Model Input Return Tensor**")
st.write(change_input)

model_output = dl_model(change_input.unsqueeze(0)) # new axis at column 0

model_output_list = model_output.squeeze(0).detach().numpy()

# Get Loaded Model Out-of-Sample Returns
# =============================================
if 'model_output_returns' not in st.session_state:
    st.session_state.model_output_returns = None

if st.session_state.upload_dl_model:
    st.session_state.model_output_returns = pm.get_returns(weights = model_output_list, 
                                        returns = out_of_sample_returns, 
                                        window = 50, 
                                        target_volatility = target_std, 
                                        use_vol_scaling = False)

model_output_returns = st.session_state.model_output_returns

# New DL Model Portfolio Weights
# =============================================
st.write("**Model Predicted Weights Output**")
order_to_market = coinbase_acc.tickers_weight(st.session_state.portfolio_tickers, model_output_list)
today = datetime.date.today()
st.dataframe(pd.DataFrame([order_to_market], columns = order_to_market.keys(), index = [today.strftime('%Y-%m-%d')]))
st.write("**Invested Amount**")
st.write(np.sum(np.abs(list(order_to_market.values()))))

# =============================================
# Model Out-of-sample Performance   
# =============================================
st.markdown("### Trading Model Validation Performance")

st.markdown(
    r"""

    We train all models on the in sample data to obtain "optimal" portfolio weights and then evaluate 
    the out-of-sample cumulative returns of all stratgeies to determine the portfolio optimisation method that yields the 
    greatest final return. For determeinistic models (MVO, MVO+BL) we calculate out-of-sample returns by investing a 
    weighted quantity of our wealth corresponfing to each asset, we then calculate the cumulative product of weighted returns
    to yield the portfolio equity curve. For DL models we predict a set of portfolio weights at each time period using
    the prior peridos returns and then invest the respective proportion of our weath in the asset. Note we assumme 
    no rebalancing costs to maintain a constant invested quantity of wealth in each asset for simplicity. 

    *To see all the equity curves (and return plots) of our models vs benchmarks ensure you have trained DmNN, DmLSTM and DLS.
    MVO, MVO+BL and the trading model are already loaded.*

    """
)

# Model Camparisons
# =============================================

keys_to_session_state(keys = ['mc_select', 'mc_bl_weights', 'mvo_eqty', 'mvo_bl_eqty', 'mvo_reset'], type = None)

mc_select_box = st.selectbox("**Market-cap Method for MVO+BL**", ["random", "yfinance", "fmp", "polygon"], index = 1)
st.session_state.mc_select = mc_select_box

# define expected return evctor using bl allocation
mvo_bl_weights_dict, _ = mvo.calculate_mvo_bl(pd.DataFrame(in_sample_returns, columns = tickers),
                                            tickers, 
                                            market_cap_method = st.session_state.mc_select, 
                                            api_key = user_api_key)

st.session_state.mc_bl_weights = mvo_bl_weights_dict
mvo_bl_weights_dict = st.session_state.mc_bl_weights

mvo_weights = np.array(list(mvo_weights_dict.values()))
mvo_bl_weights = np.array(list(mvo_bl_weights_dict.values()))

# get out-of-sample returns 
out_of_sample_returns_mvo = pm.get_returns(mvo_weights, out_of_sample_returns, window = volatility_scaling_window, \
                                        target_volatility = target_std, use_vol_scaling = False)
out_of_sample_returns_mvo_bl = pm.get_returns(mvo_bl_weights, out_of_sample_returns, window = volatility_scaling_window, \
                                            target_volatility = target_std, use_vol_scaling = False)

# get equity curves and plot performance
st.session_state.mvo_eqty = pm.equity_curve(out_of_sample_returns_mvo)
st.session_state.mvo_bl_eqty = pm.equity_curve(out_of_sample_returns_mvo_bl)

# Plots 
# =============================================
if model_output_returns is not None: 
    out_sample_dl_returns = pd.DataFrame({'DmNN': dmnn_out_sample_returns, 
                                          'DmLSTM': dmlstm_out_sample_returns, 
                                          'DLS': np.pad(dls_out_sample_returns, (len(out_of_sample_lagged_returns) - len(dls_out_sample_returns), 0)),
                                          'Imported Trading Model': model_output_returns, 
                                          'MVO': out_of_sample_returns_mvo, 
                                          'MVO+BL': out_of_sample_returns_mvo_bl}) # create dataframe for returns 
    out_sample_model_equities = pd.DataFrame({
        'DmNN': pm.equity_curve(out_sample_dl_returns['DmNN']),
        'DmLSTM': pm.equity_curve(out_sample_dl_returns['DmLSTM']),
        'DLS': pm.equity_curve(out_sample_dl_returns['DLS']),
        'Imported Trading Model': pm.equity_curve(out_sample_dl_returns['Imported Trading Model']),
        'MVO': st.session_state.mvo_eqty, 
        'MVO+BL': st.session_state.mvo_bl_eqty
    })
    # Calculate equity curves
    st.line_chart(out_sample_dl_returns, x_label = 'time', y_label = 'returns')
    st.line_chart(out_sample_model_equities, x_label = 'time', y_label = 'cumulative returns')

# =============================================
# Send / Close Market Orders
# =============================================
st.markdown("### 🚀 Send Market Orders: Portfolio Creation and Rebalancing")

st.markdown(
    """

    Click the button below once to send a market order that either creates or rebalances a portfolio. This button will 
    invest in assets listed under "Coinbase Cryptocurrency Assets" heading in the "Trading via Coinbase" section, investing
    a proportion of your account wealth equal to the DL model weightes output (displayed in a dataframe under the heading
    "Model Predicted Weights Output")

    """
)

if 'full_close' not in st.session_state: 
    st.session_state.full_close = False    
if 'order_sent' not in st.session_state:
    st.session_state.order_sent = False    

keys_to_session_state(keys = ['order_message', 'orders_closed_message', 'weight_diffs'], type = None)

# Send orders
# =============================================
# Button that triggers the order
st.button('➡️ Send Portfolio Order @ Market', key='portfolio_order_button')

if st.session_state.portfolio_order_button:
    if not st.session_state.order_sent: # check orders not sent 
        # Send order
        order = coinbase_acc.multi_asset_invest(
            portfolio_ticker_weights=order_to_market, 
            account_base='GBP', 
            full_close=False
        )

        st.session_state.order_sent = True
        st.session_state.order_message = order['orders'] # from None 
        st.session_state.weight_diffs = order['weight_diffs']
        
# Display result if order was sent
if st.session_state.order_message is not None:
    st.write("**Coinbase Order Message**", st.session_state.order_message)

    st.markdown(
        r"""
        If the order message, "Coinbase Order Message", returns a $\verb|None|$ type object, a $\verb|{success: False, ...}|$ 
        dictionary or an empty array then the portfolio market order was not successfull. 
        """
    )

    # print weight diffs 
    st.write("**Portfolio Weight Changes: Weight Δ**")
    st.write(st.session_state.weight_diffs)

    st.markdown(
        """
        if the order message, "Portfolio Weight Changes: Weight Δ", returns an empty list, then we jave just created a portfolio.
        """
    )

# Full Close Portfolio
# =============================================
st.markdown("### Delete Portfolio")

st.markdown(
    """
    The button below will close all open market positions, selling or returning the value of all crypto assets in our
    portfolio. The value of all closed positions is credited to our base account.
    """
)

st.button('⬅️ Close All Positions @ Market', key = 'full_close_button')

if st.session_state.full_close_button:
    if not st.session_state.full_close: # check orders not closed 
        # close all positions
        order_close_all = coinbase_acc.multi_asset_invest(
            portfolio_ticker_weights=order_to_market, 
            account_base='GBP', 
            full_close=True
        )
        
        st.session_state.full_close = True 
        st.session_state.orders_closed_message = order_close_all['orders']
        st.session_state.weight_diffs = order_close_all['weight_diffs']

if st.session_state.orders_closed_message is not None:
    st.write("**Orders Closed Message**", st.session_state.orders_closed_message)

    st.markdown(
        """
        If "Orders Closed Message" returns an empty list then their where no orders (of specified assets in "
        Coinbase Cryptocurrency Assets") to close. 
        """
    )

# Reset orders
# =============================================
st.markdown("### **Clear Orders**")

st.markdown(
    """
    To send new market orders or close all open and filled positions click the button below.
    """
)

st.button('🔄 Clear Order Log', key = 'clear_orders_button')

if st.session_state.clear_orders_button:
    st.session_state.order_sent = False
    st.session_state.full_close = False    
    st.session_state.order_message = None
    st.session_state.orders_closed_message = None    
    st.session_state.current_weight = None