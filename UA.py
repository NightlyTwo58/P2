import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_data(period='3y', interval='1d'):
    ticker = yf.Ticker('UAL')
    df = ticker.history(period=period, interval=interval)
    df = df[['Close', 'Volume']].dropna()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    return df

def prepare_lstm_data(df, feature_cols=['Close', 'Volume', 'MA10', 'MA50'], lookback=20):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, feature_cols.index('Close')])  # predict scaled close
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def build_and_train_lstm(X, y, epochs, batch_size, patience, checkpoint_path):
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    earlystop_cb = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    return model, history

def forecast_lstm(model, df, scaler, lookback=20, steps=30, feature_cols=['Close', 'Volume', 'MA10', 'MA50']):
    # Use last sequence to bootstrap forecasts iteratively
    recent = df[feature_cols].iloc[-lookback:].copy()
    scaled_recent = scaler.transform(recent)
    pred_scaled = []
    current_sequence = scaled_recent.copy()
    for _ in range(steps):
        inp = current_sequence.reshape((1, lookback, len(feature_cols)))
        next_scaled_close = model.predict(inp, verbose=0)[0, 0]
        next_row = current_sequence[-1].copy()
        next_row[feature_cols.index('Close')] = next_scaled_close
        current_sequence = np.vstack([current_sequence[1:], next_row])
        pred_scaled.append(next_scaled_close)
    close_idx = feature_cols.index('Close')
    dummy = np.zeros((len(pred_scaled), len(feature_cols)))
    dummy[:, close_idx] = pred_scaled
    close_min = scaler.data_min_[close_idx]
    close_max = scaler.data_max_[close_idx]
    pred_close = np.array(pred_scaled) * (close_max - close_min) + close_min
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)  # business days
    return pd.Series(pred_close, index=future_dates, name='LSTM_Forecast')

def load_lstm(checkpoint_path, X):
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.load_weights(checkpoint_path)
    return model


def run_prophet(df, periods=30):
    prophet_df = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods, freq='b')  # business days
    forecast = m.predict(future)
    forecast_series = forecast.set_index('ds')['yhat']
    return m, forecast_series

def plot_results(df, lstm_forecast, prophet_series):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='Historical Close', marker='o', linewidth=1)

    last_hist_date = df.index[-1]
    prophet_fit = prophet_series.loc[:last_hist_date]
    prophet_forecast = prophet_series.loc[prophet_series.index > last_hist_date]

    plt.plot(prophet_fit, label='Prophet Fit (in-sample)', linestyle=':', alpha=0.7)
    plt.plot(prophet_forecast, label='Prophet Forecast', linestyle='-.')
    plt.plot(lstm_forecast, label='LSTM Forecast', linestyle='--')
    plt.title('UAL Stock Close: Historical and Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_lstm(df, model, scaler, lookback=20, feature_cols=['Close', 'Volume', 'MA10', 'MA50']):
    # Use last portion of historical data as walk-forward test (simple in-sample evaluation)
    X, y, _ = prepare_lstm_data(df, feature_cols=feature_cols, lookback=lookback)
    preds_scaled = model.predict(X, verbose=0).flatten()
    close_idx = feature_cols.index('Close')
    close_min = scaler.data_min_[close_idx]
    close_max = scaler.data_max_[close_idx]
    preds = preds_scaled * (close_max - close_min) + close_min
    actual_scaled = y
    actual = actual_scaled * (close_max - close_min) + close_min
    mse = mean_squared_error(actual, preds)
    return mse, actual, preds

def main():
    df = get_data(period='5y')
    print(df)
    df.index = df.index.tz_localize(None).normalize()
    print(df)
    feature_cols = ['Close', 'Volume', 'MA10', 'MA50']
    lookback = 20  # days

    # Prepare LSTM data
    X, y, scaler = prepare_lstm_data(df, feature_cols=feature_cols, lookback=lookback)
    lstm_model, history = build_and_train_lstm(X, y, 80, 8, 10, "checkpoints/lstm_best.weights.h5")

    # Evaluate LSTM on historical
    mse, actual, preds = evaluate_lstm(df, lstm_model, scaler, lookback=lookback, feature_cols=feature_cols)
    print(f"LSTM in-sample MSE (close price): {mse:.4f}")

    # Forecast future steps days with LSTM
    lstm_forecast = forecast_lstm(lstm_model, df, scaler, lookback=lookback, steps=60, feature_cols=feature_cols)

    prophet_model, prophet_forecast = run_prophet(df, periods=30)

    plot_results(df, lstm_forecast, prophet_forecast)

    # Optionally, show side-by-side last few actual vs predicted for LSTM
    last_dates = df.index[lookback:]
    lstm_eval_df = pd.DataFrame({
        'Actual_Close': actual,
        'LSTM_Predicted_Close': preds
    }, index=last_dates)
    print("Recent in-sample comparison (last 10):")
    print(lstm_eval_df.tail(10))

if __name__ == '__main__':
    main()
