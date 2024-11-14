import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

# Generate sample data for demonstration
dates = pd.date_range(start='2017-01-01', end='2024-01-01', freq='B')  # Using 'B' for business days
np.random.seed(42)
base_price = 100
price = [base_price]
for i in range(len(dates)-1):
    change = np.random.normal(0, 2)
    new_price = price[-1] * (1 + change/100)
    price.append(new_price)

# Create DataFrame with explicit frequency
df = pd.DataFrame({
    'Date': dates,
    'Close': price
})
df.set_index('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index).to_period('B')  # Set business day frequency

def calculate_moving_averages(data):
    """Calculate various moving averages"""
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['MA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
    data['MA365'] = data['Close'].rolling(window=365, min_periods=1).mean()
    data['MA500'] = data['Close'].rolling(window=500, min_periods=1).mean()
    return data

def test_stationarity(data):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(data)
    return result[0], result[1]

def find_optimal_order(data, max_p=5, max_d=2, max_q=5):
    """Find optimal ARIMA order based on AIC"""
    best_aic = float('inf')
    best_order = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order

# Calculate moving averages
df = calculate_moving_averages(df)

# Convert PeriodIndex to DatetimeIndex for plotting
df.index = df.index.to_timestamp()

# Test for stationarity
stationary_test_stat, stationary_p_value = test_stationarity(df['Close'])

# Find optimal ARIMA order
optimal_order = find_optimal_order(df['Close'])

# Fit ARIMA model with optimal order
model = ARIMA(df['Close'], order=optimal_order, freq='B')
results = model.fit()

# Calculate autocorrelations
acf_values = acf(df['Close'], nlags=40, fft=False)

# Create figure for plotting
plt.figure(figsize=(15, 10))

# Price and Moving Averages
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Close'], label='Price', alpha=0.6)
plt.plot(df.index, df['MA50'], label='50-day MA', alpha=0.8)
plt.plot(df.index, df['MA200'], label='200-day MA', alpha=0.8)
plt.plot(df.index, df['MA365'], label='365-day MA', alpha=0.8)
plt.plot(df.index, df['MA500'], label='500-day MA', alpha=0.8)
plt.title('TATA MOTORS Stock Price with Moving Averages')
plt.legend()

# Autocorrelation Plot
plt.subplot(2, 1, 2)
plt.bar(range(len(acf_values)), acf_values)
plt.title('Autocorrelation Function')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle='--', color='gray')

plt.tight_layout()

# Print summary statistics
print(f"""
ARIMA Model Analysis Summary:
----------------------------
Optimal ARIMA Order: {optimal_order}
Stationarity Test:
- ADF Test Statistic: {stationary_test_stat:.4f}
- p-value: {stationary_p_value:.4f}

Moving Averages (Latest Values):
- 50-day MA: {df['MA50'].iloc[-1]:.2f}
- 200-day MA: {df['MA200'].iloc[-1]:.2f}
- 365-day MA: {df['MA365'].iloc[-1]:.2f}
- 500-day MA: {df['MA500'].iloc[-1]:.2f}

Model Summary:
{results.summary().tables[1]}
""")