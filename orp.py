import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the tickers for the stocks
tickers = ['AAPL', 'GOOGL', 'META', 'TSLA']

# Download the stock price data (adjusted close)
data = yf.download(tickers, start='2015-01-01', end='2024-12-31')['Close']

# Calculate the daily returns
returns = data.pct_change().dropna()

# Calculate expected returns and covariance matrix
expected_returns = returns.mean() * 252  # Annualize the mean returns (252 trading days)
cov_matrix = returns.cov() * 252  # Annualize the covariance matrix

# Number of assets
num_assets = len(tickers)

# Function to calculate portfolio return
def portfolio_return(weights):
    return np.dot(weights, expected_returns)

# Function to calculate portfolio volatility (risk)
def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Function to calculate the negative Sharpe ratio
def negative_sharpe_ratio(weights, risk_free_rate=0.0):
    ret = portfolio_return(weights)
    vol = portfolio_volatility(weights)
    return -(ret - risk_free_rate) / vol  # Negative Sharpe ratio for minimization

# Constraints: Sum of weights should be 1, weights should be between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess for weights (equal allocation)
initial_weights = [1. / num_assets] * num_assets

# Minimize the negative Sharpe ratio
opt_result = minimize(negative_sharpe_ratio, initial_weights, bounds=bounds, constraints=constraints)

# Optimal weights for the risky portfolio
optimal_weights = opt_result.x

# Calculate the expected return and volatility of the optimal portfolio
optimal_return = portfolio_return(optimal_weights)
optimal_volatility = portfolio_volatility(optimal_weights)

# Print the results
print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

print("\nOptimal Portfolio Return:", optimal_return)
print("Optimal Portfolio Volatility:", optimal_volatility)

# Plotting the efficient frontier and the optimal risky portfolio
def efficient_frontier():
    target_returns = np.linspace(0, 0.5, 100)
    target_volatilities = []
    
    for target_return in target_returns:
        def objective(weights):
            return portfolio_volatility(weights)
        
        constraints = ({'type': 'eq', 'fun': lambda weights: portfolio_return(weights) - target_return},
                       {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        target_volatilities.append(result.fun)
    
    return target_returns, target_volatilities

target_returns, target_volatilities = efficient_frontier()

# Plot the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.plot(target_volatilities, target_returns, label='Efficient Frontier')
plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=100, label='Optimal Risky Portfolio')
plt.title('Efficient Frontier and Optimal Risky Portfolio')
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.legend(loc='upper left')
plt.show()
