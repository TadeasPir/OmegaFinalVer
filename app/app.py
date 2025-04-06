from flask import Flask, request, jsonify, render_template
import os
import pickle
import pandas as pd
import yaml
import logging
import logging.config

app = Flask(__name__)

# Define the base directory and load the configuration from config.yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

try:
    with open(CONFIG_PATH, "r") as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    raise Exception("Error loading configuration: " + str(e))

# Set up logging configuration from the YAML config if available
if "logging" in config:
    logging.config.dictConfig(config["logging"])
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("Configuration loaded successfully.")

# Load your model as usual...
MODEL_PATH = os.path.join(BASE_DIR, "models", "gradientboosting_best_model.pkl")
logger.info("Loading model from: %s", MODEL_PATH)

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading the model: %s", e)
    model = None

# Define expected features
expected_features = [
    "Volume",
    "MA5",
    "MA20",
    "MA50",
    "RSI",
    "BB_Middle",
    "BB_Std",
    "BB_Upper",
    "BB_Lower",
    "EMA12",
    "EMA26",
    "MACD",
    "MACD_Signal",
    "Daily_Return",
    "Volatility_14d",
    "marketCap",
    "neg",
    "neu",
    "pos",
]

# ---------- Define Routes ----------
@app.route("/")
def home():
    logger.info("Rendering home page")
    # This will render the index.html file located in the templates folder
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received prediction request")
    if model is None:
        logger.error("Model not loaded.")
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(force=True)
    try:
        if isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            input_df = pd.DataFrame([data])

        missing_features = set(expected_features) - set(input_df.columns)
        if missing_features:
            logger.error("Missing features: %s", list(missing_features))
            return jsonify({
                "error": "Missing features in input data",
                "missing_features": list(missing_features)
            }), 400

        input_df = input_df[expected_features]
        prediction = model.predict(input_df)
        predictions_list = prediction.tolist()
        logger.info("Prediction successful")
        return jsonify({"prediction": predictions_list})
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Retrieve Flask run configuration from YAML under the "flask" section
    flask_config = config.get("flask",)
    debug = flask_config.get("debug")
    host = flask_config.get("host")
    port = flask_config.get("port")
    logger.info("Starting Flask app on %s:%s, debug=%s", host, port, debug)
    app.run(debug=debug, host=host, port=port)
