import os
from pathlib import Path


class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent.parent

    # Model path
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'models', 'gb_model.pkl'))

    # API settings
    API_TITLE = 'Financial Market Prediction API'
    API_VERSION = 'v1'

    # Debug settings
    DEBUG = os.environ.get('DEBUG', 'False') == 'True'
