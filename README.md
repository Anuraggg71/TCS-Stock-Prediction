# TCS Stock Data – Live and Latest

> Internship Project | Submission Date: 25/06/2025  
> By Anurag Dewangan

---

## 🎯 Objective

Analyze and predict TCS stock closing prices using:
- Machine Learning: Linear Regression  
- Deep Learning: LSTM (Long Short-Term Memory)

---

## 📁 Project Structure

| File                             | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| `TCS_LSTM_Prediction.ipynb`     | Full LSTM model + EDA and all visualizations                |
| `TCS_stock_history.csv`         | Historical stock data from 2002 to 2024                     |
| `tcs_lstm_model.h5`             | Trained LSTM model (optional)                               |
| `predictions.csv`               | LSTM predicted results (optional)                           |
| `README.md`                     | Project summary and documentation                           |

---

## 📊 Visualizations Included

| Graph Title                              | Purpose                                                  |
|------------------------------------------|----------------------------------------------------------|
| **Close Price Over Time**                | Shows price trend throughout history                     |
| **Volume, Dividends, Stock Splits**      | Understand stock actions over time                       |
| **Correlation Heatmap**                  | Relationship between features like Open, High, Low       |
| **30-Day Moving Average**                | Smooth out price trends                                  |
| **Moving Average Crossover Strategy**    | Simulated buy/sell signals                               |
| **Daily % Price Change Histogram**       | Visualize volatility and return distribution             |
| **Actual vs Predicted (LSTM)**           | LSTM performance on last 500 days                        |
| **Prediction Line on Future Price**      | Shows the predicted price visually compared to history   |

---

## 📈 Dataset Overview

- **Source:** Provided via Google Drive  
- **Time Range:** 2002 – 2024  
- **Columns:** Date, Open, High, Low, Close, Volume, Dividends, Stock Splits  
- **Total Records:** 4,463

---

## 🤖 Models Used

### 1. Linear Regression (ML)
- Features: Open, High, Low, Volume, Prev_Close, Day_of_Week, Month  
- R² Score: ~0.999  
- Output: Close price prediction + Actual vs Predicted Scatter Plot

### 2. LSTM (Deep Learning)
- Sequence Length: 60 days  
- Architecture: 50 LSTM units + Dense  
- Output: 1-day close price forecast  
- Example Prediction: ₹3861.72 on next date

---

## ⚙️ Technologies Used

- Python 3.11  
- Jupyter Notebook  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`

---

## ✅ Summary

- Complete pipeline from data cleaning → EDA → modeling → prediction  
- LSTM adds deep learning capability beyond simple regression  
- Project documented with clean visual outputs for analysis and demo  
- Fully GitHub-deployable and extendable for future forecasting

---

### 📌 Project Conclusion

The TCS stock has demonstrated a strong and steady growth trajectory over the past two decades, supported by solid fundamentals and market leadership in the IT services sector. Despite short-term fluctuations, the overall long-term trend is bullish, which aligns with the company's financial strength and global expansion.

Using both Linear Regression and LSTM, we successfully modeled the stock’s behavior, with the LSTM model effectively capturing temporal patterns for short-term forecasting. These models can be used to further analyze investor opportunities or develop real-time trading strategies.

This project not only reflects the technical capability of modern prediction models but also underlines the strength and stability of TCS as a key player in the Indian stock market.

---

## 📬 Contact

For queries or contributions, feel free to reach me:  
📧 **anuragdewangan1209@gmail.com**
