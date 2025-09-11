import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import DLS_model as dls 
import DmNN as dmnn

# =============================================
# New RNN Model for Portfolio Weights
# =============================================

class PortfolioRNN(nn.Module):
    """
    Recurrent Neural Network (RNN) for generating portfolio weights based on sequential asset returns.
    
    This model uses an LSTM architecture to process time-series data of asset returns, capturing temporal
    dependencies. It is designed to predict weights for the current time step based on prior returns,
    avoiding information leakage from future data.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=100, num_layers=2, long_short=True):
        super().__init__()
        self.long_short = long_short
        self.L_activation = nn.Tanh() if self.long_short else nn.Softmax(dim=-1)
        
        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Final linear layer to map LSTM output to weights
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass.
        """
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        
        raw_weights = self.fc(out)  # (batch_size, seq_len, output_dim)
        
        if self.long_short:
            activated = self.L_activation(raw_weights)
            sum_abs = torch.sum(torch.abs(activated), dim=-1, keepdim=True)
            output = activated / (sum_abs + 1e-8)  # Avoid division by zero
        else:
            output = self.L_activation(raw_weights)
        
        return output

# =============================================
# Adapted Backpropagation for RNN Model
# =============================================

def backprop_rnn(asset_returns, target_returns, target_volatility, hidden_dim=100, num_layers=2, 
                 use_vol_scaling=True, long_short=True, distance_metric='mmd', n_epochs=1000, 
                 learning_rate=0.001):
    """
    Train PortfolioRNN with backpropagation, avoiding information leakage.
    """
    m, n_assets = asset_returns.shape
    if m < 2:
        raise ValueError("Insufficient time steps for sequential processing.")
    
    model = PortfolioRNN(input_dim=n_assets, output_dim=n_assets, 
                         hidden_dim=hidden_dim, num_layers=num_layers, long_short=long_short)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Prepare sequences to avoid leakage
    X_prior = torch.FloatTensor(asset_returns).unsqueeze(0)  # (1, m-1, n_assets)
    X_current = torch.FloatTensor(asset_returns)  # (m-1, n_assets)
    R_p = torch.FloatTensor(target_returns)  # (m-1,)
    
    # Compute EMA std on training data (prior to avoid leakage)
    ema_std = dls.std_ema(asset_returns)
    sigma_ratio = target_volatility / ema_std
    sigma_ratio_tensor = torch.FloatTensor(sigma_ratio)
    
    loss_history = []  # For monitoring convergence (publication quality)
    portfolio_original_weights = None
    portfolio_returns_initial = None
    final_weights = None
    portfolio_returns_final = None

    for epoch in tqdm(range(n_epochs)):
        weights = model(X_prior)  # (1, m-1, n_assets)
        weights = weights.squeeze(0)  # (m-1, n_assets)
        
        # Compute portfolio returns
        scaled_returns = X_current * (sigma_ratio_tensor if use_vol_scaling else torch.tensor(1.0))
        R = torch.sum(weights * scaled_returns, dim=1)  # (m-1,)
        
        if epoch == 0:
            portfolio_original_weights = weights.detach().numpy()
            portfolio_returns_initial = R.detach().numpy()
        
        # Compute loss
        if distance_metric == "l2":
            loss = dmnn.l2_loss(R, R_p)  # Assuming l2_loss from DmNN
        elif distance_metric == "mmd":
            loss = dmnn.mmd_unbiased(R.unsqueeze(1), R_p.unsqueeze(1))  # Adjust to 2D if needed
        elif distance_metric == "wasserstein":
            loss = dmnn.wasserstein_distance(R, R_p)
        else:
            raise ValueError("Invalid distance_metric")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())  # Track for plots
    
    # After training, compute average weights
    with torch.no_grad():
        weights = model(X_prior)
        weights = weights.squeeze(0).detach().numpy()  # (m-1, n_assets)
        avg_weights = weights.mean(axis=0)   # average across time steps
        
        if long_short:
            final_weights = avg_weights / np.sum(np.abs(avg_weights) + 1e-8)
        else:
            final_weights = avg_weights 
        
        # Final portfolio returns
        scaled_returns = X_current * (sigma_ratio_tensor if use_vol_scaling else torch.tensor(1.0))
        portfolio_returns_final = torch.sum(torch.FloatTensor(final_weights) * scaled_returns, dim=1).detach().numpy()

    # For publication: Return loss history for plotting convergence
    return (portfolio_original_weights, portfolio_returns_initial, weights, final_weights, 
            portfolio_returns_final, loss_history, model)