import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import DLS_model as dls # import DLS model for volatility scaling 
import streamlit as st 

# =============================================
# DmNN
# =============================================

class PortfolioNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, neurons_per_layer=100, long_short=True):
        super().__init__() #  allows to to access parent class attributes of nn.Modules __init__() method, we can also access nn.Modules class methods
        self.long_short = long_short
        self.L_activation = nn.Tanh() if self.long_short else nn.Softmax(dim=-1)
        self.input_dim = input_dim
        self.output_dim = output_dim 

        # Dynamic architecture based on hyperparameters
        layers = []
        layers.append(nn.Linear(self.input_dim, neurons_per_layer))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(neurons_per_layer, self.output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        raw_weights = self.fc(x)
        if self.long_short:
            L_activated = self.L_activation(raw_weights)
            output = L_activated / torch.sum(torch.abs(L_activated), dim=1, keepdim=True)
        else:
            output = self.L_activation(raw_weights)
        return output
   
# =============================================
# Loss Function: MMD (Unbiased Estimator) Squared 
# =============================================

# MMD loss function
def gaussian_kernel(X, Y, sigma=1.0):  # the RBF kernel 
    # Convert to tensors if inputs are numpy arrays
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(Y, np.ndarray):
        Y = torch.FloatTensor(Y)
        
    X = X.unsqueeze(1)  # Shape (m, 1, d)
    Y = Y.unsqueeze(0)  # Shape (1, n, d)
    squared_dist = torch.sum((X - Y) ** 2, dim=-1)  # Shape (m, n)
    return torch.exp(-squared_dist / (2 * sigma ** 2))

# Unbiased empirical MMD^2 estimator gradient wrt return vector
def mmd_unbiased(R, R_p): 
    m, n = R.shape[0], R_p.shape[0]
    K_RR = gaussian_kernel(R, R)
    K_RRp = gaussian_kernel(R, R_p)
    
    # Term 1: (Sum of off-diagonal K_RR) / (m(m-1))
    sum_K_RR = torch.sum(K_RR) - torch.trace(K_RR)
    term1 = sum_K_RR / (m * (m - 1))
    
    # Term 2: (Sum of K_RRp) * 2 / (mn)
    sum_K_RRp = torch.sum(K_RRp)
    term2 = 2 * sum_K_RRp / (m * n)
    
    return term1 - term2

# =============================================
# Loss Function: L2^2 Divergance 
# =============================================

# kernel functions that does not detach from compuation graph 
def kde(u: torch.Tensor) -> torch.Tensor:
    # Use torch constants to preserve gradients and device/dtype
    const = 1.0 / torch.sqrt(torch.tensor(2) * torch.pi)
    u = torch.as_tensor(u, dtype=torch.float32)
    return const * torch.exp(torch.tensor(-0.5) * u**2)

def l2_loss(X, Y):
    return torch.norm(kde(X) - kde(Y)) * (torch.max(X) - torch.min(X))/len(X)  # L2 norm

# =============================================
# Loss Functions: Wasserstein Distance  
# =============================================

def wasserstein_distance(X, Y):
    # get the order statistics
    X_sort = torch.sort(torch.FloatTensor(X), dim = 0)[0]
    Y_sort = torch.sort(torch.FloatTensor(Y), dim = 0)[0]
    return torch.sum((X_sort - Y_sort)**2)**0.5   # Mean absolute difference

# =============================================
# Training Progress Function 
# =============================================
def train_progress(epoch, epochs, loss_function, show):
    if show: 
        try:
            import streamlit as st
            if (epoch % 10 == 0):
                st.write(f"⏳ Training progress | {(epoch / epochs) * 100}% | Loss = {loss_function.item()}") 
            if (epoch == epochs - 1):
                st.write(f"✅ Model Trained / Training Progress | 100% | Loss = {loss_function.item()}")
        except Exception as e: 
            print("Falied to import streamlit, check package is installed")
    else: 
        pass

# =============================================
# DmNN Backpropagation Algorithm
# =============================================
def backprop(asset_returns, target_returns, target_volatility, hidden_layers=2, neurons_per_layer=100, 
     use_vol_scaling = True, long_short = True, distance_metric='mmd', n_epochs = 1000, show_progress = False):
    _, n_assets = asset_returns.shape
    model = PortfolioNN(input_dim=n_assets, output_dim=n_assets, long_short = long_short)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Convert numpy arrays to PyTorch tensors
    X = torch.FloatTensor(asset_returns) 
    R_p = torch.FloatTensor(target_returns)
    
    # Compute 50-day EMA of standard deviations for each asset
    # First calculate daily returns standard deviation with a 50-day rolling window
    numpy_returns = asset_returns.numpy() if torch.is_tensor(asset_returns) else asset_returns
    
    # Calculate EMA of standard deviations
    ema_std = dls.std_ema(asset_returns)
    
    sigma_ratio = target_volatility / ema_std
    sigma_ratio_tensor = torch.FloatTensor(sigma_ratio)

    loss = None
    portfolio_original_weights = None
    portfolio_returns_initial = None
    final_weights = None

    for epoch in tqdm(range(n_epochs)):
        
        # Forward pass: Get weights and compute portfolio returns
        weights = model(X)  # Shape (m, n_assets)
        # Modified return calculation according to the formula (**NO REBALANCING ASSUMPTION BEING MADE**)
        R = torch.sum(weights * X * (sigma_ratio_tensor if use_vol_scaling else torch.tensor(1.0)), dim=1, keepdim=True)  # dim = 1 = column sum => Shape (m, 1)
        
        # Save initial returns for comparison
        if epoch == 0:
            portfolio_original_weights = weights.detach().numpy()
            portfolio_returns_initial = R.detach().numpy()

        # Compute loss gradients via auto grad
        if distance_metric == "l2": 
            loss = l2_loss(R, R_p)
        if distance_metric == "mmd":
            loss = mmd_unbiased(R, R_p)
        elif distance_metric == "wasserstein":
            loss = wasserstein_distance(R, R_p)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        # model progress report 
        train_progress(epoch, n_epochs, loss, show_progress)

    # After training, compute average weights
    with torch.no_grad():
        X = torch.FloatTensor(asset_returns) # To tensor 
        weights = model(X).detach().numpy()  # shape (m, n_assets)
       
        # if want to use non dynamic portfolio optimisation (use set of optimal weights from training only)
        avg_weights = weights.mean(axis=0)   # average across samples
        # Renormalize to maintain constraint
        if long_short:
            # enforce sum of absolute weights = 1
            final_weights = avg_weights / np.sum(np.abs(avg_weights))
        else:
            # enforce sum of weights = 1
            final_weights = avg_weights 
        
    # If you want to save the final portfolio returns as well
    portfolio_returns_final = R.detach().numpy()

    return portfolio_original_weights, portfolio_returns_initial, weights, final_weights, portfolio_returns_final, model