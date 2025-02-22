import joblib
import pandas as pd
import numpy as np
from io import StringIO

def evaluate_differential_privacy(noise_multiplier=1.1, batch_size=64, dataset_size=10000, epochs=10):
    """
    Evaluate differential privacy parameters using TensorFlow Privacy.
    Computes epsilon using the DP-SGD accountant.
    """
    try:
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
    except ImportError as e:
        return {"error": f"TensorFlow Privacy not installed: {str(e)}"}
    
    epsilon, _ = compute_dp_sgd_privacy(
        dataset_size, batch_size, noise_multiplier, epochs, delta=1e-5
    )
    return epsilon

def perform_privacy_tests(model_content: bytes, train_content: bytes):
    """
    The privacy test that performs a basic membership inference attack and
    estimates differential privacy parameters.
    
    Expects:
      - model_content: a pickled model file.
      - train_content: a CSV file containing training data.
    """
    try:
        model = joblib.loads(model_content)
    except Exception as e:
        return {"error": f"Failed to load model for privacy test: {str(e)}"}
    
    try:
        data_str = train_content.decode("utf-8")
        train_df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse training CSV: {str(e)}"}
    
    if 'label' in train_df.columns:
        X_train = train_df.drop(columns=['label'])
    else:
        X_train = train_df.copy()
    
    try:
        train_probs = model.predict_proba(X_train)
        train_confidences = np.max(train_probs, axis=1)
    except Exception as e:
        return {"error": f"Model does not support predict_proba: {str(e)}"}
    
    synthetic_data = X_train.copy()
    synthetic_data = synthetic_data.apply(lambda col: col + np.random.normal(0, 0.1, size=len(col)))
    
    synthetic_probs = model.predict_proba(synthetic_data)
    synthetic_confidences = np.max(synthetic_probs, axis=1)
    
    threshold = np.mean(synthetic_confidences)
    
    member_predictions = (train_confidences >= threshold).astype(int)
    non_member_predictions = (synthetic_confidences >= threshold).astype(int)
    
    train_accuracy = np.mean(member_predictions == 1)
    non_member_accuracy = np.mean(non_member_predictions == 0)
    attack_accuracy = 0.5 * (train_accuracy + non_member_accuracy)
    
    # Use TensorFlow Privacy to compute epsilon
    epsilon = evaluate_differential_privacy(
        noise_multiplier=1.1,
        batch_size=64,
        dataset_size=len(X_train),
        epochs=10
    )
    
    return {
        "membership_inference_attack_accuracy": attack_accuracy,
        "threshold": threshold,
        "train_member_confidences_mean": float(np.mean(train_confidences)),
        "synthetic_confidences_mean": float(np.mean(synthetic_confidences)),
        "differential_privacy_epsilon": epsilon,
        "note": "Lower attack accuracy and higher epsilon indicate better privacy protection."
    }
