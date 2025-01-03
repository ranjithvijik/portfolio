import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fredapi import Fred
import time

# Initialize FRED API (you need an API key)
fred_api_key = '724ad7fc925d199d9dd1a64ff5bcd71a'  # Replace with your API Key
fred = Fred(api_key=fred_api_key)

# DJIA ticker symbols (last updated as of 2023)
djia_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'DIS', 'V', 'INTC', 'CRM', 'CSCO', 'PYPL', 'KO', 'JNJ', 'WMT', 'PFE',
    'MCD', 'CVX', 'HD', 'BA', 'CAT', 'IBM', 'GS', 'TRV', 'UNH', 'MS', 'MMM', 'WBA', 'KO', 'GE', 'XOM', 'ABT'
]

# Function to download stock price data
def download_stock_data(tickers, start_date, end_date):
    try:
        stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        if stock_data.empty:
            raise ValueError("Stock data is empty.")
        return stock_data
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return None

# Function to download macroeconomic data from FRED
def download_macro_data(start_date, end_date):
    try:
        # Example: GDP, Inflation (CPI), Interest Rate (10-year Treasury yield), and market returns (S&P 500)
        gdp = fred.get_series('GDP', observation_start=start_date, observation_end=end_date)
        cpi = fred.get_series('CPIAUCSL', observation_start=start_date, observation_end=end_date)
        interest_rate = fred.get_series('GS10', observation_start=start_date, observation_end=end_date)
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close'].squeeze() # added .squeeze() here
        
        # Resample data to monthly frequency (end of month)
        gdp = gdp.resample('ME').last()  # Using last available data for month
        cpi = cpi.resample('ME').last()
        interest_rate = interest_rate.resample('ME').last()
        sp500 = sp500.resample('ME').last()
        
        # Align data
        macro_data = pd.DataFrame({'GDP': gdp, 'Inflation': cpi, 'Interest Rate': interest_rate, 'Market': sp500.pct_change()})
        macro_data = macro_data.dropna()
        
        if macro_data.empty:
            raise ValueError("Macroeconomic data is empty.")
        return macro_data
    except Exception as e:
        print(f"Error downloading macro data: {e}")
        return None

# Function to calculate factor betas for each stock
def calculate_betas(stock_data, macro_data):
    betas = {}
    for ticker in stock_data.columns:
        stock_returns = stock_data[ticker].pct_change().dropna()
        
        # Align stock returns and macro data by index
        aligned_data = pd.concat([stock_returns, macro_data], axis=1, join='inner')
        
        if aligned_data.empty:
            print(f"No overlapping data for {ticker}, skipping.")
            continue
        
        X = aligned_data[['Market', 'Interest Rate', 'Inflation', 'GDP']]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = aligned_data[ticker]
        
        try:
            model = sm.OLS(y, X).fit()
            betas[ticker] = model.params[1:]  # Skip the constant (alpha)
        except Exception as e:
            print(f"Error calculating betas for {ticker}: {e}")
            betas[ticker] = [np.nan] * 4  # Return NaNs if regression fails
    
    return pd.DataFrame(betas).T

# Function to calculate expected returns based on betas, risk premiums and risk free rate
def calculate_expected_returns(betas_df, risk_premiums, risk_free_rate, djia_tickers):
  expected_returns = {}
  for ticker in djia_tickers:
    if ticker not in betas_df.index:
      print(f"No beta data for {ticker}, skipping.")
      continue
    betas_for_stock = betas_df.loc[ticker]
    expected_return = risk_free_rate  # Start with risk-free rate 
      
    for factor, beta in betas_for_stock.items():
      expected_return += beta * risk_premiums.get(factor, 0)  # Use 0 if factor not in risk premiums
    
    expected_returns[ticker] = expected_return
  return pd.DataFrame(expected_returns, index=['Expected Return']).T

# Function to plot expected returns
def plot_expected_returns(expected_returns_df):
    plt.figure(figsize=(12, 6))
    plt.bar(expected_returns_df.index, expected_returns_df['Expected Return'], color='skyblue')
    plt.axhline(y=expected_returns_df['Expected Return'].mean(), color='r', linestyle='--', label='Average Expected Return')
    plt.xlabel('DJIA Stocks')
    plt.ylabel('Expected Return')
    plt.title('Expected Returns for DJIA Stocks using APT Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# Define risk premiums and risk-free rate
def calculate_risk_premiums(macro_data):
  # Calculate historical risk premiums (e.g., for market)
  market_return = macro_data['Market'].mean() - macro_data['Market'].shift(1).mean()
  interest_rate = macro_data['Interest Rate'].mean()
  inflation = macro_data['Inflation'].mean()
  gdp_growth = macro_data['GDP'].mean()
  
  # Risk premium assumptions (based on averages or custom assumptions)
  return {
      'Market': market_return,
      'Interest Rate': interest_rate,
      'Inflation': inflation,
      'GDP': gdp_growth
  }

def main():
  # Define the date range
  start_date = '2010-01-01'
  end_date = '2024-12-31'
    
  # Download stock and macro data
  stock_data = download_stock_data(djia_tickers, start_date, end_date)
  if stock_data is None:
    return
  macro_data = download_macro_data(start_date, end_date)
  if macro_data is None:
    return
    
  # Calculate betas
  betas_df = calculate_betas(stock_data, macro_data)
  if betas_df.empty:
    print("No betas calculated, cannot proceed")
    return

  # Call this function before calculating expected returns
  risk_premiums = calculate_risk_premiums(macro_data)
  risk_free_rate = 0.045  # 4.5% risk-free rate (e.g., 10-year treasury yield)
    
  # Calculate expected returns for each stock using APT
  expected_returns_df = calculate_expected_returns(betas_df, risk_premiums, risk_free_rate, djia_tickers)
    
  # Display the expected returns
  print(expected_returns_df)
  
  plot_expected_returns(expected_returns_df)

# Run the main analysis
if __name__ == "__main__":
    main()

def validate_model(expected_returns_df, stock_data, validation_period='1Y'):
    # Calculate actual returns for validation period
    validation_end_date = pd.to_datetime(stock_data.index[-1])
    validation_start_date = validation_end_date - pd.DateOffset(years=1)
    validation_data = stock_data.loc[validation_start_date:validation_end_date]
    
    actual_returns = validation_data.pct_change().mean()
    
    # Compare expected returns with actual returns
    comparison = expected_returns_df.loc[:, 'Expected Return'] - actual_returns
    print("Comparison between expected and actual returns (for 1-year validation):")
    print(comparison)
    
    # Calculate correlation
    correlation = expected_returns_df['Expected Return'].corr(actual_returns)
    print(f"Correlation between expected and actual returns: {correlation:.2f}")
