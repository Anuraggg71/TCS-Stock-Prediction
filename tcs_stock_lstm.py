# 📦 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import warnings
warnings.filterwarnings("ignore")

# 📥 Load the Dataset
df = pd.read_csv('TCS_stock_history.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.reset_index(drop=True, inplace=True)

# ✅ BASIC STATS
print(df.info())
print(df.describe())

# 🔁 Normalize Close Prices for LSTM
close_data = df[['Date', 'Close']].copy()
scaler = MinMaxScaler()
close_data['Close_Scaled'] = scaler.fit_transform(close_data[['Close']])

# ============================
# 📊 EXPLORATORY DATA ANALYSIS
# ============================

# 📈 1. Close Price Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.title("TCS Stock Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 📊 2. Volume, Dividends, Stock Splits Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Volume'], label='Volume', color='green')
plt.plot(df['Date'], df['Dividends'], label='Dividends', color='red')
plt.plot(df['Date'], df['Stock Splits'], label='Stock Splits', color='purple')
plt.title("Volume, Dividends, and Stock Splits Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='Greens', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 📈 4. 30-Day Moving Average
df['30D_MA'] = df['Close'].rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.plot(df['Date'], df['30D_MA'], label='30-Day MA', color='orange')
plt.title("Close Price with 30-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📈 5. Moving Average Crossover Strategy
df['Short_MA'] = df['Close'].rolling(window=5).mean()
df['Long_MA'] = df['Close'].rolling(window=30).mean()
df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 1, -1)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.plot(df['Date'], df['Short_MA'], label='5-Day MA', color='red')
plt.plot(df['Date'], df['Long_MA'], label='30-Day MA', color='green')
plt.scatter(df['Date'], df['Close'] * df['Signal'], label='Buy/Sell Signal', color='magenta', marker='o', alpha=0.3)
plt.title("Moving Average Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📉 6. Daily Price Change Histogram
df['Daily_Change_%'] = df['Close'].pct_change() * 100
plt.figure(figsize=(8, 5))
sns.histplot(df['Daily_Change_%'].dropna(), kde=True, color='orange')
plt.title("Distribution of Daily % Price Change")
plt.xlabel("Daily % Change")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================
# 🤖 LSTM MODELING
# ============================

# 🧠 Create LSTM Sequences (60-day lookback)
sequence_length = 60
X_lstm = []
y_lstm = []
scaled_close = close_data['Close_Scaled'].values

for i in range(sequence_length, len(scaled_close)):
    X_lstm.append(scaled_close[i-sequence_length:i])
    y_lstm.append(scaled_close[i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# 🧱 Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 🏋️ Train the Model
model.fit(X_lstm, y_lstm, epochs=20, batch_size=32)

# 🔮 Predict Next Close Price
last_60_days = scaled_close[-60:]
X_test = np.reshape(last_60_days, (1, 60, 1))
predicted_scaled = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_scaled)

# 📅 Show Prediction
last_date = close_data['Date'].iloc[-1]
next_date = last_date + pd.Timedelta(days=1)
print(f"\n📅 Predicted Date: {next_date.strftime('%Y-%m-%d')}")
print(f"📈 Predicted Close Price: ₹{predicted_price[0][0]:.2f}")

# ============================
# 📈 VISUALIZE PREDICTION
# ============================

# Plot Last 500 Days + Predicted Price
recent_data = close_data[-500:]

plt.figure(figsize=(12, 6))
plt.plot(recent_data['Date'], recent_data['Close'], label="Historical Close Prices", color='blue')
plt.axhline(y=predicted_price[0][0], color='red', linestyle='--',
            label=f"Predicted ₹{predicted_price[0][0]:.2f} on {next_date.strftime('%Y-%m-%d')}")
plt.title("TCS Stock Price + LSTM Prediction (Last 500 Days)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 💾 Save the LSTM Model (Optional)
model.save("tcs_lstm_model.h5")
