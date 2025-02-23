import pickle

def validate_model(file_bytes: bytes):
    """
    Validates the uploaded model file by attempting to unpickle it.
    Returns a tuple: (is_valid: bool, message: str, model: object or None)
    """
    try:
        model = pickle.loads(file_bytes)
        return (True, "Model file is valid.", model)
    except Exception as e:
        return (False, f"Error loading model: {e}", None)

def get_model_info(model: object):
    """
    Extracts basic information about the model.
    For demonstration, returns the model's type.
    """
    info = {
        "model_type": str(type(model))
    }
    return info
