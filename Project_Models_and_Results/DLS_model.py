import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from Project_Models_and_Results.Performance_Metrics import calculate_sharpe_ratio
import DmNN as dmnn

'''
DLS Model attempted replication from [10]
'''

# =============================================
# LSTM RNN agenet 
# =============================================
class SharpeLSTM(nn.Module):
    def __init__(self, n_assets: int, lookback: int, lstm_units: int = 64):
        super().__init__()
        self.n_assets = n_assets
        self.lstm = nn.LSTM(input_size = 2 * n_assets,
                             hidden_size = lstm_units, 
                             batch_first = True)
        self.output = nn.Linear(lstm_units, n_assets)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x) # output dimiensions = (batch_size, sequence_length, hidden_dimensions) = (n_assets, t, 64)
        last = lstm_out[:, -1, :]

        return self.softmax(self.output(last))  # long-only: weights ≥ 0, sum to 1

# =============================================
#  Standard Deviation Based EMA 
# =============================================

def std_ema(x, window=50, eps=1e-8):
    """
    Calculate the exponential moving average of standard deviation for each column in x.
    x: numpy array or tensor of shape (n_timesteps, n_assets)
    window: integer, the lookback period for the EMA
    eps: small constant to avoid division by zero
    """
    x = x.numpy() if torch.is_tensor(x) else x  # Convert tensor to numpy array
    n_timesteps, n_assets = x.shape
    ema_std = np.zeros_like(x, dtype=float)
    
    # Initialize the first value with the standard deviation of the initial window
    if window >= n_timesteps:
        init_std = np.std(x, axis=0, ddof=1)  # Use sample standard deviation
    else:
        init_std = np.std(x[:window], axis=0, ddof=1)
    ema_std[0] = init_std
    
    # Calculate alpha for the EMA (half-life approximation)
    alpha = 2 / (window + 1)
    
    # Compute EMA of variance and then take square root for standard deviation
    ema_var = np.zeros_like(x, dtype=float)
    ema_var[0] = init_std**2  # Initial variance
    
    for t in range(1, n_timesteps):
        # Use the maximum available data up to the current time for the window
        start_idx = max(0, t - window)
        window_data = x[start_idx:t + 1]
        if len(window_data) > 1:
            current_var = np.var(window_data, axis=0, ddof=1)
        else:
            current_var = ema_var[t-1]  # Fallback to previous variance if window is too small
        ema_var[t] = alpha * current_var + (1 - alpha) * ema_var[t-1]
        ema_std[t] = np.sqrt(ema_var[t] + eps)  # Add epsilon for numerical stability
    
    return ema_std

# =============================================
#  REMOVE and import from Performance_Metrics.py 
# =============================================

def portfolio_returns(weights: torch.Tensor, future_returns: torch.Tensor, target_volatility, window, use_vol_scaling = True) -> torch.Tensor:
    """Compute portfolio return:"""
    ema_std = std_ema(future_returns, window)
    sigma_ratio = target_volatility / ema_std
    sigma_ratio_tensor = torch.FloatTensor(sigma_ratio)
    return torch.sum(weights * future_returns * (sigma_ratio_tensor if use_vol_scaling else torch.tensor(1.0)), dim=1) #***include Transaction Costs***#

# =============================================
#  LSTM Batch data 
# =============================================

def prepare_features(asset_returns: np.ndarray, asset_prices: np.ndarray, lookback: int):
    n_timesteps, n_assets = asset_returns.shape
    X, Y = [], []

    # Lag features and get lagged fearture set with observations from lookback starting from lookback (first feature entry) up to current index
    # i.e. get lagged variables set, lagged by lookback
    for t in range(lookback, n_timesteps):
        past_prices = asset_prices[t - lookback:t] # will get n-timesteps - lookback number of batches
        past_returns = asset_returns[t - lookback:t]
        features = np.concatenate([past_prices, past_returns], axis=1) # each batch is a of size n_assets*2 
        X.append(features)
        Y.append(asset_returns[t]) # the return is our label

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32) # pass arrays transform to tensors, ready for model input 

# =============================================
#  DLS Backpropagation Algorithm 
# =============================================

def train_model(asset_returns: np.ndarray, 
                asset_prices: np.ndarray, 
                target_volatility,
                window = 50, 
                lookback: int = 50,
                n_epochs: int = 100, 
                lr: float = 0.001,
                batch_size: int = 64, 
                vol_scaling = True, 
                show_progress = False) -> np.ndarray:
    # prepare featurs
    features, targets = prepare_features(asset_returns, asset_prices, lookback)
    n_assets = asset_returns.shape[1]
    # instentiate model and optimizer
    model = SharpeLSTM(n_assets, lookback)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_samples = features.shape[0] # number of rows in n_obs * n_assets np array 

    # backprop
    for epoch in tqdm(range(n_epochs)):
        indices = torch.randperm(n_samples) # random permutation of integers equal to size of input - 1 
        # for each epoch calculate batch updates
        for i in range(0, n_samples, batch_size): # batch here is the skip per index
            idx = indices[i: i + batch_size] # index permutation of integers
            X_batch, y_batch = features[idx], targets[idx] # get randoom permutation of features and targets

            weights = model(X_batch)
            port_ret = portfolio_returns(weights, y_batch, target_volatility, window, use_vol_scaling = vol_scaling) # change to sharp calculated in pm 
            #loss = calculate_sharpe_ratio(port_ret)
            loss = (torch.mean(port_ret, dim = 0)) / torch.std(port_ret, dim = 0) # maximise sharpe ratio 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # print progress
        dmnn.train_progress(epoch, n_epochs, loss, show_progress)

    # disable gradinet tracking 
    with torch.no_grad(): # set grad to false, only for model evaluation / prediction. This is required True for model training 
        final_input = features[-1:].unsqueeze(0) if features.ndim == 2 else features[-1:]
        final_weights = model(final_input)
        
        return final_weights.squeeze(0).numpy(), model