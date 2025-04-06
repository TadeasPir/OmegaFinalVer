from flask import Flask, request, jsonify, render_template
import os
import pickle
import pandas as pd

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
    raise("Error loading the model:", e)
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
    app.run(debug=True, host="127.0.0.1", port=5000)
