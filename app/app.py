from flask import Flask, request, jsonify, render_template
import os
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
app = Flask(__name__)

# Load your model as usual...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "gradientboosting_best_model.pkl")
print("Loading model from:", model_path)

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)
    model = None

# Define expected features
expected_features = [
    "Volume", "MA5", "MA20", "MA50", "RSI",
    "BB_Middle", "BB_Std", "BB_Upper", "BB_Lower",
    "EMA12", "EMA26", "MACD", "MACD_Signal",
    "Daily_Return", "Volatility_14d", "marketCap",
    "neg", "neu", "pos"
]

# ---------- Define Routes ----------
@app.route("/")
def home():
    # This will render the index.html file located in the templates folder
    return render_template("index.html")


@app.route("/auto_predict")
def auto_predict():
    return render_template("auto_predict.html")


@app.route("/auto_predict_api", methods=["POST"])
def auto_predict_api():
    """
    API endpoint for automated predictions using yfinance data
    """
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(force=True)
    ticker = data.get("ticker", "AAPL")  # Default to AAPL

    try:
        # Fetch historical data using yfinance
        stock = yf.Ticker(ticker)

        # Get historical data for the past 60 days (we need extra days for calculating indicators)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        historical_data = stock.history(start=start_date, end=end_date)

        if historical_data.empty:
            return jsonify({"error": f"No data available for {ticker}"}), 404

        # Calculate technical indicators
        df = historical_data.copy()

        # Calculate Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

        # Calculate MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate Daily Return and Volatility
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility_14d'] = df['Daily_Return'].rolling(window=14).std()

        # Get market cap (can be None if not available)
        try:
            market_cap = stock.info.get('marketCap', 0)
        except:
            market_cap = 0

        # Add dummy sentiment values (would normally come from sentiment analysis)
        df['neg'] = 0.2
        df['neu'] = 0.5
        df['pos'] = 0.3

        # Drop NaN values
        df = df.dropna()

        # For prediction, use the last available data point
        input_data = df.iloc[-1:].copy()
        input_data['marketCap'] = market_cap

        # Ensure all expected features are present
        missing_features = set(expected_features) - set(input_data.columns)
        if missing_features:
            return jsonify({
                "error": "Missing features in input data",
                "missing_features": list(missing_features)
            }), 400

        # Select only the features the model expects
        input_data = input_data[expected_features]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get confidence (this will depend on your model)
        # For demonstration, if using a model with predict_proba
        try:
            confidence = float(model.predict_proba(input_data)[0][prediction])
        except:
            confidence = 0.8  # Fallback value

        # Format data for the frontend
        # Last 30 days for chart
        chart_data = historical_data.tail(30).copy()
        dates = [d.strftime('%b %d') for d in chart_data.index]
        prices = chart_data['Close'].tolist()

        # Generate future dates
        future_dates = []
        future_prices = []
        last_price = prices[-1]

        # Simple linear projection based on prediction
        if prediction == 1:  # Up trend
            slope = 0.005  # 0.5% daily increase
        else:
            slope = -0.005  # 0.5% daily decrease

        for i in range(1, 8):  # 7 days ahead
            future_date = (end_date + timedelta(days=i)).strftime('%b %d')
            future_dates.append(future_date)

            # Project price with some randomness
            future_price = last_price * (1 + slope * i + np.random.normal(0, 0.005))
            future_prices.append(future_price)

        # Prepare response
        response = {
            "ticker": ticker,
            "lastUpdated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "currentPrice": float(historical_data['Close'].iloc[-1]),
            "priceChange": float(historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]),
            "priceChangePercent": float(
                ((historical_data['Close'].iloc[-1] / historical_data['Close'].iloc[-2]) - 1) * 100),
            "prediction": int(prediction),
            "confidence": float(confidence),
            "priceHistory": {
                "dates": dates,
                "prices": prices
            },
            "predictedPrices": {
                "dates": future_dates,
                "prices": future_prices
            },
            "indicators": {
                "rsi": float(df['RSI'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "bb": "Near upper band" if df['Close'].iloc[-1] > df['BB_Middle'].iloc[-1] else "Near lower band",
                "volume": float(df['Volume'].iloc[-1]),
                "volatility": float(df['Volatility_14d'].iloc[-1])
            },
            "sentiment": {
                "positive": 0.3,
                "neutral": 0.5,
                "negative": 0.2
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(force=True)
    try:
        if isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            input_df = pd.DataFrame([data])

        missing_features = set(expected_features) - set(input_df.columns)
        if missing_features:
            return (
                jsonify({
                    "error": "Missing features in input data",
                    "missing_features": list(missing_features)
                }),
                400,
            )

        input_df = input_df[expected_features]
        prediction = model.predict(input_df)
        predictions_list = prediction.tolist()
        return jsonify({"prediction": predictions_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
