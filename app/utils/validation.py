def validate_input_data(data):
    """Validate input data for prediction"""
    if not data:
        return {
            'status': 'error',
            'message': 'No data provided'
        }

    if 'data' not in data:
        return {
            'status': 'error',
            'message': 'Input must contain a "data" field with records'
        }

    if not isinstance(data['data'], list) or len(data['data']) == 0:
        return {
            'status': 'error',
            'message': 'The "data" field must be a non-empty array of records'
        }

    return {
        'status': 'success'
    }
