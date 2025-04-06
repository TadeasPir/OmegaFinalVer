
# ML Prediction

This Flask application provides an API endpoint to predict outcomes using a pre-trained Gradient Boosting model. The app has a homepage wiht UI for predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
  - [Home Endpoint](#home-endpoint)
  - [Predict Endpoint](#predict-endpoint)
- [Error Handling](#error-handling)
- [License](#license)

## Prerequisites

- Python 3.x  
- Required Python packages:
  - `Flask`
  - `pandas`
  -  `PyYAML`

- Ensure that you have your trained model file:  
  `models/gradientboosting_best_model.pkl`

## Installation

1. **Clone the Repository**  
   Clone the repository (or copy the code) into your local machine.

2. **Set Up a Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv venv
   venv\Scripts\activate # For Linux/Mac: source venv/Scripts/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Directory Structure**  
   Ensure your project structure looks similar to:
   ```plaintext
   project_directory/
   ├── app.py
   ├── config.yaml
   ├── models/
   │   └── gradientboosting_best_model.pkl
   └── templates/
       └── index.html
   ```

## Usage

Run the Flask application by executing:

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000` with debug mode enabled.
can be changed in config.yaml
example config:
```yaml
flask:
  debug: true
  host: "127.0.0.1"
  port: 5000

logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    simple:
      format: "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
  handlers:
    console:
      class: logging.StreamHandler
      level: DEBUG
      formatter: simple
      stream: ext://sys.stdout
  root:
    level: INFO
    handlers: [console]

```

## API Endpoints

### Home Endpoint

- **URL:** `/`
- **Description:**  
  Renders and returns the `index.html` template found in the `templates` folder.

### Predict Endpoint

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Description:**  
  Accepts JSON data, validates the input, and returns the prediction made by the loaded model.

- **Expected Input Features:**  
  The JSON payload must include all of the following features:
  - `Volume`
  - `MA5`
  - `MA20`
  - `MA50`
  - `RSI`
  - `BB_Middle`
  - `BB_Std`
  - `BB_Upper`
  - `BB_Lower`
  - `EMA12`
  - `EMA26`
  - `MACD`
  - `MACD_Signal`
  - `Daily_Return`
  - `Volatility_14d`
  - `marketCap`
  - `neg`
  - `neu`
  - `pos`

  The payload can either be a single JSON object or a list of objects.

- **Example Request (Single Object):**
  ```json
  {
    "Volume": 1000000,
    "MA5": 102.5,
    "MA20": 101.2,
    "MA50": 99.8,
    "RSI": 55,
    "BB_Middle": 100,
    "BB_Std": 5,
    "BB_Upper": 105,
    "BB_Lower": 95,
    "EMA12": 101.0,
    "EMA26": 100.0,
    "MACD": 1.0,
    "MACD_Signal": 0.8,
    "Daily_Return": 0.02,
    "Volatility_14d": 0.03,
    "marketCap": 5000000000,
    "neg": 0.2,
    "neu": 0.5,
    "pos": 0.3
  }
  ```

- **Example Response:**
  ```json
  {
    "prediction": [predicted_value]
  }
  ```
  If multiple input objects are provided, the response will be a list of predictions.

- **Error Responses:**
  - **Missing Model:**  
    If the model is not loaded, the API returns:
    ```json
    {
      "error": "Model not loaded."
    }
    ```
  - **Missing Features:**  
    If any expected feature is missing from the input:
    ```json
    {
      "error": "Missing features in input data",
      "missing_features": ["MissingFeature1", "MissingFeature2"]
    }
    ```
  - **General Errors:**  
    For any unexpected errors during prediction:
    ```json
    {
      "error": "Error message describing the failure"
    }
    ```

## Error Handling

- **Model Loading Error:**  
  If the model file fails to load at application startup, the `/predict` endpoint will return an error message indicating that the model is missing.
  
- **Input Validation:**  
  The API verifies that all expected features are present. If any are missing, it returns a 400 status code with details of the missing features.
  

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
