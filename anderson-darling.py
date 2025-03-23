# -*- coding: utf-8 -*-

"""
Created on Sun Mar 23 20:52:18 2025

@author: Hemal
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_returns(price_data, frequency='daily'):
    """
    Calculate returns from price data based on specified frequency
    """
    freq_map = {
        'daily': 'B',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }
    freq = freq_map.get(frequency.lower(), 'B')

    if frequency.lower() == 'daily':
        returns = price_data['Close'].pct_change().dropna()
    else:
        resampled_prices = price_data['Close'].resample(freq).last()
        returns = resampled_prices.pct_change().dropna()

    return returns

def perform_anderson_darling_test(returns):
    """
    Perform Anderson-Darling test for normality
    """
    returns_array = np.array(returns).flatten()
    result = stats.anderson(returns_array, dist='norm')
    statistic = result.statistic
    critical_values = result.critical_values
    significance_levels = [15, 10, 5, 2.5, 1]

    st.write("Anderson-Darling Test Statistic: {:.4f}".format(statistic))
    st.write("Critical Values:")
    for i, (cv, sl) in enumerate(zip(critical_values, significance_levels)):
        st.write("  {}%: {:.4f}".format(sl, cv))

    if statistic > critical_values[2]:
        st.write("\nAt 5% significance level, we reject the null hypothesis.")
        st.write("The returns do NOT follow a normal distribution.")
    else:
        st.write("\nAt 5% significance level, we fail to reject the null hypothesis.")
        st.write("The returns may follow a normal distribution.")

    return statistic, critical_values, significance_levels

def visualize_returns(returns, frequency):
    """
    Create visualizations for the returns data
    """
    returns_values = np.array(returns).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Returns time series
    axes[0, 0].plot(returns.index, returns_values)
    axes[0, 0].set_title(f'{frequency.capitalize()} Returns Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Returns')

    # Plot 2: Histogram with normal distribution overlay
    axes[0, 1].hist(returns_values, bins=30, density=True, alpha=0.7)
    x = np.linspace(min(returns_values), max(returns_values), 100)
    y = stats.norm.pdf(x, np.mean(returns_values), np.std(returns_values))
    axes[0, 1].plot(x, y, 'r-', lw=2)
    axes[0, 1].set_title(f'Histogram of {frequency.capitalize()} Returns with Normal Curve')
    axes[0, 1].set_xlabel('Returns')

    # Plot 3: Q-Q plot
    stats.probplot(returns_values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')

    # Plot 4: Box plot
    axes[1, 1].boxplot(returns_values)
    axes[1, 1].set_title(f'Box Plot of {frequency.capitalize()} Returns')

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Stock Returns Analysis")

    # User inputs
    ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL):")
    start_date = st.date_input("Enter start date:", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Enter end date:", pd.to_datetime("2023-01-01"))

    # Get return frequency from user
    frequency = st.selectbox("Select return calculation frequency:",
                             ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
    frequency = frequency.lower()

    if st.button("Analyze"):
        try:
            # Fetch data
            st.write(f"Fetching data for {ticker}...")
            stock_data = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if stock_data.empty:
                st.write(f"No data found for {ticker}. Please check the ticker symbol.")
                return

            # Calculate returns with the selected frequency
            returns = calculate_returns(stock_data, frequency)

            if len(returns) == 0:
                st.write("No returns could be calculated. Check if there's enough price data.")
                return

            st.write(f"Calculated {len(returns)} {frequency} returns from {returns.index.min().date()} to {returns.index.max().date()}")

            # Calculate basic statistics
            st.write(f"Basic statistics of {frequency} returns:")
            returns_array = np.array(returns).flatten()
            mean_val = np.mean(returns_array)
            std_val = np.std(returns_array)
            skew_val = stats.skew(returns_array)
            kurt_val = stats.kurtosis(returns_array)

            st.write(f"Mean: {mean_val:.6f}")
            st.write(f"Standard Deviation: {std_val:.6f}")
            st.write(f"Skewness: {skew_val:.6f}")
            st.write(f"Kurtosis: {kurt_val:.6f}")

            # Perform Anderson-Darling test
            st.write(f"Performing Anderson-Darling test for normality on {frequency} returns:")
            perform_anderson_darling_test(returns)

            # Visualize the returns
            st.write(f"Generating visualizations for {frequency} returns...")
            visualize_returns(returns, frequency)

        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
            st.write(f"Error details: {type(e).__name__}")

if __name__ == "__main__":
    main()
