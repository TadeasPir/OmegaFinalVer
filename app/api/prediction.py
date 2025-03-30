import pandas as pd
import pickle
import os
from config.settings import MODEL_PATH
from app.utils.validation import validate_input_data


def make_prediction(data):
    """Make prediction using the pre-trained model"""

    # Validate input data
    validation_result = validate_input_data(data)
    if validation_result['status'] == 'error':
        return validation_result

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return {
            'status': 'error',
            'message': 'Model not found. Please ensure the pre-trained model exists.'
        }

    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model_info = pickle.load(f)

    pipeline = model_info['pipeline']
    features = model_info['features']
    model_type = model_info['model_type']

    # Convert to DataFrame
    df = pd.DataFrame(data['data'])

    # Check for missing features
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        return {
            'status': 'error',
            'message': f'Missing features: {", ".join(missing_features)}'
        }

    # Make predictions
    predictions = pipeline.predict(df[features]).tolist()

    result = {
        'status': 'success',
        'predictions': predictions
    }

    # For classification, include probabilities
    if model_type == 'classification' and hasattr(pipeline, 'predict_proba'):
        probabilities = pipeline.predict_proba(df[features]).tolist()
        result['probabilities'] = probabilities

    return result


def get_model_info():
    """Get information about the pre-trained model"""

    if not os.path.exists(MODEL_PATH):
        return {
            'status': 'error',
            'message': 'Model not found. Please ensure the pre-trained model exists.'
        }

    with open(MODEL_PATH, 'rb') as f:
        model_info = pickle.load(f)

    return {
        'status': 'success',
        'features': model_info['features'],
        'target': model_info['target'],
        'model_type': model_info['model_type'],
        'categorical_features': model_info.get('categorical_features', []),
        'numerical_features': model_info.get('numerical_features', [])
    }
