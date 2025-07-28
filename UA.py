import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for numerical operations


def get_stock_data(ticker_symbol, period):
    """
    Fetches daily stock prices and resamples them to quarterly averages.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'UAL').
        period (str): The period for which to fetch data (e.g., '5y').

    Returns:
        tuple: A tuple containing:
            - yf.Ticker object
            - pandas Series of quarterly average close prices
            - pandas DataFrame of daily prices
    """
    ticker = yf.Ticker(ticker_symbol)
    daily_prices = ticker.history(period=period)
    # Resample daily close prices to quarterly end (December) mean
    quarterly_close = daily_prices['Close'].resample('QE-DEC').mean()
    return ticker, quarterly_close, daily_prices


def get_financials(ticker, target_index):
    """
    Fetches quarterly financial statements (income, balance, cash flow)
    and aligns their indexes with the target index.

    Args:
        ticker (yf.Ticker): The yfinance Ticker object.
        target_index (pd.DatetimeIndex): The index to align financial data with.

    Returns:
        tuple: A tuple containing:
            - pandas DataFrame of quarterly income statement
            - pandas DataFrame of quarterly balance sheet
            - pandas DataFrame of quarterly cash flow statement
    """
    def align(df_to_align):
        # Transpose the DataFrame to have dates as index
        df_to_align = df_to_align.T
        # Convert index to datetime objects
        df_to_align.index = pd.to_datetime(df_to_align.index)
        # Filter to include only dates present in the target_index
        return df_to_align[df_to_align.index.isin(target_index)]

    income_q = align(ticker.quarterly_financials)
    balance_q = align(ticker.quarterly_balance_sheet)
    cashflow_q = align(ticker.quarterly_cashflow)
    return income_q, balance_q, cashflow_q


def get_shares_outstanding(ticker, index, daily_prices):
    """
    Fetches shares outstanding data and resamples to quarterly.
    Provides a fallback if full shares data is not available.

    Args:
        ticker (yf.Ticker): The yfinance Ticker object.
        index (pd.DatetimeIndex): The target index for the shares data.
        daily_prices (pd.DataFrame): DataFrame of daily prices to determine start/end dates.

    Returns:
        pd.Series: Quarterly shares outstanding.
    """
    try:
        shares = ticker.get_shares_full(start=daily_prices.index.min(), end=daily_prices.index.max())
        return shares['Shares Outstanding'].resample('Q').last()
    except Exception as e:
        print(f"Warning: Could not fetch full shares outstanding data. Error: {e}")
        fallback = ticker.info.get('sharesOutstanding', None)
        # If fallback is None, it will create a Series of NaNs
        return pd.Series([fallback] * len(index), index=index)


def compute_metrics(close, income_q, balance_q, cashflow_q, shares_outstanding):
    """
    Creates a DataFrame of metrics by directly including existing financial stats.
    No new ratios or derived metrics are calculated in this version.

    Args:
        close (pd.Series): Quarterly average close prices.
        income_q (pd.DataFrame): Quarterly income statement.
        balance_q (pd.DataFrame): Quarterly balance sheet.
        cashflow_q (pd.DataFrame): Quarterly cash flow statement.
        shares_outstanding (pd.Series): Quarterly shares outstanding.

    Returns:
        pd.DataFrame: DataFrame containing selected existing financial stats and stock price.
    """
    # Ensure all input indexes are timezone-naive for consistent merging/alignment
    close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
    income_q.index = income_q.index.tz_localize(None) if income_q.index.tz is not None else income_q.index
    balance_q.index = balance_q.index.tz_localize(None) if balance_q.index.tz is not None else balance_q.index
    cashflow_q.index = cashflow_q.index.tz_localize(None) if cashflow_q.index.tz is not None else cashflow_q.index
    shares_outstanding.index = shares_outstanding.index.tz_localize(
        None) if shares_outstanding.index.tz is not None else shares_outstanding.index

    # Initialize the metrics DataFrame with the stock price
    df = pd.DataFrame(index=close.index)
    df['Avg_Stock_Price'] = close

    print("Columns in balance_q:", balance_q.columns)
    print("Columns in income_q:", income_q.columns) # Also print income_q columns for reference
    print("Columns in cashflow_q:", cashflow_q.columns) # Also print cashflow_q columns for reference

    # --- Directly add selected existing financial stats ---
    # Using .get() with pd.NA as fallback to avoid KeyError if a column is missing
    # and allow .dropna() to handle missing values later in the main function.

    # From Income Statement
    df['Total Revenue'] = income_q.get('Total Revenue', pd.NA)
    df['Net Income'] = income_q.get('Net Income', pd.NA)
    df['Gross Profit'] = income_q.get('Gross Profit', pd.NA)
    df['Operating Income'] = income_q.get('Operating Income', pd.NA)
    df['EBIT'] = income_q.get('EBIT', pd.NA)
    df['EBITDA'] = income_q.get('EBITDA', pd.NA)
    df['Interest Expense'] = income_q.get('Interest Expense', pd.NA)
    df['Tax Provision'] = income_q.get('Tax Provision', pd.NA)

    # From Balance Sheet
    df['Total Assets'] = balance_q.get('Total Assets', pd.NA)
    df['Total Liabilities Net Minority Interest'] = balance_q.get('Total Liabilities Net Minority Interest', pd.NA)
    df['Stockholders Equity'] = balance_q.get('Stockholders Equity', pd.NA)
    df['Total Current Assets'] = balance_q.get('Total Current Assets', pd.NA)
    df['Total Current Liabilities'] = balance_q.get('Total Current Liabilities', pd.NA)
    df['Cash And Cash Equivalents'] = balance_q.get('Cash And Cash Equivalents', pd.NA)
    df['Long Term Debt'] = balance_q.get('Long Term Debt', pd.NA)
    df['Inventory'] = balance_q.get('Inventory', pd.NA)
    df['Accounts Receivable'] = balance_q.get('Accounts Receivable', pd.NA)
    df['Property Plant And Equipment'] = balance_q.get('Property Plant And Equipment', pd.NA) # Often 'Net PPE' or similar
    df['Goodwill'] = balance_q.get('Goodwill', pd.NA)

    # From Cash Flow Statement
    df['Operating Cash Flow'] = cashflow_q.get('Operating Cash Flow', pd.NA)
    df['Investing Cash Flow'] = cashflow_q.get('Investing Cash Flow', pd.NA)
    df['Financing Cash Flow'] = cashflow_q.get('Financing Cash Flow', pd.NA)
    df['Free Cash Flow'] = cashflow_q.get('Free Cash Flow', pd.NA) # Often calculated, but sometimes directly provided

    # Add Shares Outstanding directly
    df['Shares Outstanding'] = shares_outstanding

    # IMPORTANT: Removed .dropna() from here. Data cleaning will be handled in main()
    # to be more flexible about which columns are required for specific steps.
    return df


def analyze_correlations(metrics_df, threshold=0.5):
    """
    Computes correlations between all metrics and the average stock price.
    Filters and sorts by absolute correlation strength.

    Args:
        metrics_df (pd.DataFrame): DataFrame of financial metrics and stock prices.
        threshold (float): Minimum absolute correlation value to include in filtered results.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame of filtered and sorted correlations with stock price.
            - pd.DataFrame of all correlations.
    """
    # Calculate correlations, ensuring only numeric columns are considered
    correlations = metrics_df.corr(numeric_only=True)
    stock_corr = correlations[['Avg_Stock_Price']].sort_values(by='Avg_Stock_Price', ascending=False)
    filtered = stock_corr[stock_corr['Avg_Stock_Price'].abs() >= threshold]
    filtered['Abs_Correlation'] = filtered['Avg_Stock_Price'].abs()
    filtered = filtered.sort_values('Abs_Correlation', ascending=False)
    return filtered, stock_corr


def train_model(metrics_df, predictors, target_column='Avg_Stock_Price'):
    """
    Trains a Linear Regression model using the specified predictors and target.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing features and target.
        predictors (list): List of column names to use as features.
        target_column (str): Name of the target column.

    Returns:
        tuple: A tuple containing:
            - trained LinearRegression model
            - numpy array of predictions
            - pandas Series of actual target values
    """
    X = metrics_df[predictors]
    y = metrics_df[target_column]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions, y


def plot_predictions(dates, actual, predicted):
    """
    Plots actual vs. predicted stock prices over time.

    Args:
        dates (pd.DatetimeIndex): Dates for the plot.
        actual (pd.Series): Actual stock prices.
        predicted (np.array or pd.Series): Predicted stock prices.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, label='Actual', marker='o')
    plt.plot(dates, predicted, label='Predicted', linestyle='--', marker='x')
    plt.xlabel('Quarter')
    plt.ylabel('Stock Price')
    plt.title('UAL Predicted vs Actual Stock Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def forecast_future_prices(model, metrics_df, predictors, n_quarters=4):
    """
    Forecasts future stock prices using the trained model and last known predictor values.

    Args:
        model (LinearRegression): Trained regression model.
        metrics_df (pd.DataFrame): DataFrame of historical metrics.
        predictors (list): List of predictor column names.
        n_quarters (int): Number of future quarters to forecast.

    Returns:
        pd.DataFrame: DataFrame with forecasted prices for future quarters.
    """
    # Use last known values for extrapolation for the predictors
    last_row = metrics_df.iloc[-1][predictors]

    # Build future dates
    last_date = metrics_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.QuarterEnd(),
                                 periods=n_quarters, freq='Q')

    # Repeat last values for future periods for the predictors
    future_X = pd.DataFrame([last_row] * n_quarters, columns=predictors, index=future_dates)

    # Predict future prices
    future_predicted_prices = model.predict(future_X)

    # Combine with existing
    forecast_df = pd.DataFrame({
        'Predicted_Price': future_predicted_prices
    }, index=future_dates)

    return forecast_df


def plot_with_uncertainty(dates, actual, predicted, model, X):
    """
    Plots actual and predicted stock prices with a 95% confidence interval
    around the predicted values for the historical data range.
    Note: Confidence intervals for future forecasts are more complex and
    require advanced time-series methods. This plot approximates for historical data.

    Args:
        dates (pd.DatetimeIndex): Dates for the plot (including historical and future).
        actual (pd.Series): Actual stock prices (can contain NaNs for future).
        predicted (pd.Series): Predicted stock prices (including historical and future).
        model (LinearRegression): Trained regression model.
        X (pd.DataFrame): Feature matrix used for predictions (historical + extrapolated future).
    """
    from scipy.stats import t
    import numpy as np
    import pandas as pd

    # Filter out None/NaN values from actual and predicted for statistical calculations
    # This ensures we only calculate residuals and CI on the training data range
    valid_actual_indices = actual.dropna().index
    actual_for_stats = actual.loc[valid_actual_indices]
    predicted_for_stats = pd.Series(model.predict(X.loc[valid_actual_indices]), index=valid_actual_indices)
    X_for_stats = X.loc[valid_actual_indices]


    residuals = actual_for_stats - predicted_for_stats
    n = len(residuals) # Number of observations in training data
    p = X_for_stats.shape[1] # Number of predictors

    if n <= p + 1: # Ensure enough data points for degrees of freedom (n - p - 1)
        print("Not enough data points to compute confidence interval reliably for historical data.")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=dates, y=actual, label='Actual Price', marker='o')
        sns.lineplot(x=dates, y=predicted, label='Predicted Price', marker='o')
        plt.title('Stock Price Prediction (Insufficient data for CI)')
        plt.xlabel('Quarter')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    dof = n - p - 1  # degrees of freedom for regression residuals
    s_err = np.sqrt(np.sum(residuals**2) / dof) # Residual standard error

    t_val = t.ppf(0.975, dof)  # t-value for 95% CI

    # Calculate confidence interval for the mean prediction (historical data)
    # This is a simplified approximation for plotting purposes.
    # A more rigorous calculation involves the hat matrix or covariance matrix of coefficients.
    # For now, we'll use a constant standard error for the CI band, which is common for visualization.
    ci_margin = t_val * s_err

    # Create Series for CI bounds, aligning with the full dates
    ci_upper = pd.Series(index=dates, dtype=float)
    ci_lower = pd.Series(index=dates, dtype=float)

    # Apply CI only to the historical data range
    ci_upper.loc[valid_actual_indices] = predicted_for_stats + ci_margin
    ci_lower.loc[valid_actual_indices] = predicted_for_stats - ci_margin


    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates, y=actual, label='Actual Price', marker='o')
    sns.lineplot(x=dates, y=predicted, label='Predicted Price', marker='o')

    # Fill between for the confidence interval, but only where data exists
    plt.fill_between(ci_upper.index, ci_lower, ci_upper, color='gray', alpha=0.3, label='95% Confidence Interval (Historical)')

    plt.title('Stock Price Prediction with Uncertainty Interval')
    plt.xlabel('Quarter')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the stock analysis workflow:
    1. Get stock data and financial statements.
    2. Compute metrics (using existing stats).
    3. Analyze correlations.
    4. Train a linear regression model.
    5. Plot historical predictions.
    6. Forecast future prices.
    7. Plot historical and forecasted prices with uncertainty.
    """
    ticker, close_prices, daily_prices = get_stock_data('UAL', '5y')
    income_q, balance_q, cashflow_q = get_financials(ticker, close_prices.index)
    shares_out = get_shares_outstanding(ticker, close_prices.index, daily_prices)

    metrics_df = compute_metrics(close_prices, income_q, balance_q, cashflow_q, shares_out)

    # Ensure there's enough data after fetching to proceed with correlation
    if metrics_df.empty:
        print("No financial data could be retrieved. Cannot proceed with analysis.")
        return

    filtered_corr, all_corr = analyze_correlations(metrics_df)

    print("Top correlated metrics:\n", filtered_corr)

    # Select predictors: drop 'Avg_Stock_Price' and ensure selected columns are numeric and exist
    # We also exclude 'Shares Outstanding' as it's often highly correlated with stock price
    # and might dominate the regression if not normalized, or if we want other financial drivers.
    # If 'Shares Outstanding' is a strong predictor, you can add it back.
    potential_predictors = filtered_corr.index.drop('Avg_Stock_Price', errors='ignore')
    predictors = [
        p for p in potential_predictors
        if p in metrics_df.columns and pd.api.types.is_numeric_dtype(metrics_df[p])
        and p != 'Shares Outstanding' # Exclude shares outstanding if you want to focus on financial performance
    ]

    if not predictors:
        print("No valid numeric predictors found after correlation analysis. Cannot train model.")
        print("Consider adjusting the correlation threshold or reviewing available financial columns.")
        return

    # Create a DataFrame specifically for training, dropping NaNs only for relevant columns
    df_for_training = metrics_df[['Avg_Stock_Price'] + predictors].dropna()

    if df_for_training.empty:
        print("No complete data points for selected predictors and target after dropping NaNs. Cannot train model.")
        return

    model, predicted_history, actual_history = train_model(df_for_training, predictors)

    r2 = r2_score(actual_history, predicted_history)
    mse = mean_squared_error(actual_history, predicted_history)
    print(f"\nModel R² Score: {r2:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")

    # Plot historical predictions using the index from df_for_training
    plot_predictions(df_for_training.index, actual_history, predicted_history)
    filtered_corr.to_csv("ual_strong_correlations.csv")

    # Forecast future prices - use the original metrics_df for last_row to ensure consistency
    forecast_df = forecast_future_prices(model, metrics_df, predictors)

    print("\nForecasted Stock Prices for Future Quarters:")
    print(forecast_df)

    # Combine historical and forecasted data for comprehensive plotting
    full_dates = df_for_training.index.append(forecast_df.index)
    full_actual = actual_history.reindex(full_dates) # Reindex actual to include future NaNs
    full_predicted = pd.Series(predicted_history, index=df_for_training.index).append(forecast_df['Predicted_Price'])
    full_predicted.name = 'Predicted_Price' # Name the series for consistent plotting

    # Plot combined historical and forecasted predictions
    plot_predictions(full_dates, full_actual, full_predicted)

    # Prepare X for uncertainty plot - extend last known predictor values into the future
    # Use metrics_df for the full range, then ffill and mean fill for robustness
    full_X_for_plotting = metrics_df[predictors].reindex(full_dates).fillna(method='ffill')
    full_X_for_plotting = full_X_for_plotting.fillna(full_X_for_plotting.mean())

    plot_with_uncertainty(full_dates, full_actual, full_predicted, model, full_X_for_plotting)


if __name__ == '__main__':
    main()
