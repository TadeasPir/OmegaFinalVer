from flask import Blueprint, request, jsonify, render_template
from app.models.prediction import make_prediction, get_model_info

api_bp = Blueprint('api', __name__)

@api_bp.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Keep your existing API routes
@api_bp.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = make_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/api/model_info', methods=['GET'])
def model_info():
    try:
        result = get_model_info()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
