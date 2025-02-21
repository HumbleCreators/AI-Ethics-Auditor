import joblib
import pandas as pd
import numpy as np
from io import StringIO

def perform_privacy_tests(model_content: bytes, train_content: bytes):
    """
    Perform a basic membership inference attack to evaluate privacy risk.
    
    Expects:
      - model_content: a pickled model file that supports predict_proba.
      - train_content: a CSV file containing the training data. 
        Optionally, the CSV may include a 'label' column, which is dropped.
    
    The function compares the model's confidence on training data (members)
    against synthetic non-member data (created by perturbing the training inputs).
    It computes an attack accuracy: the closer to 1, the more vulnerable the model is.
    
    Returns a dictionary with:
      - membership_inference_attack_accuracy: averaged accuracy over members and non-members.
      - threshold: the chosen confidence threshold.
      - train_member_confidences_mean: average confidence on training samples.
      - synthetic_confidences_mean: average confidence on perturbed samples.
      - note: guidance about the result.
    """
    try:
        model = joblib.loads(model_content)
    except Exception as e:
        return {"error": f"Failed to load model for privacy test: {str(e)}"}
    
    try:
        # Load training data from CSV
        data_str = train_content.decode("utf-8")
        train_df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse training CSV: {str(e)}"}
    
    # Drop the label column if present
    if 'label' in train_df.columns:
        X_train = train_df.drop(columns=['label'])
    else:
        X_train = train_df.copy()
    
    # Ensure the model supports predict_proba
    try:
        train_probs = model.predict_proba(X_train)
        train_confidences = np.max(train_probs, axis=1)
    except Exception as e:
        return {"error": f"Model does not support predict_proba: {str(e)}"}
    
    # Generate synthetic non-member data by adding Gaussian noise
    synthetic_data = X_train.copy()
    synthetic_data = synthetic_data.apply(lambda col: col + np.random.normal(0, 0.1, size=len(col)))
    
    synthetic_probs = model.predict_proba(synthetic_data)
    synthetic_confidences = np.max(synthetic_probs, axis=1)
    
    # Use the mean of synthetic confidences as the threshold
    threshold = np.mean(synthetic_confidences)
    
    # For training data, we expect the model to be more confident (member: label 1)
    member_predictions = (train_confidences >= threshold).astype(int)
    # For synthetic data, we expect lower confidence (non-member: label 0)
    non_member_predictions = (synthetic_confidences >= threshold).astype(int)
    
    # Compute accuracy: For training, correct if predicted member (1)
    train_accuracy = np.mean(member_predictions == 1)
    # For synthetic, correct if predicted non-member (0)
    non_member_accuracy = np.mean(non_member_predictions == 0)
    
    # Attack accuracy: average of both (lower attack accuracy indicates better privacy)
    attack_accuracy = 0.5 * (train_accuracy + non_member_accuracy)
    
    return {
        "membership_inference_attack_accuracy": attack_accuracy,
        "threshold": threshold,
        "train_member_confidences_mean": float(np.mean(train_confidences)),
        "synthetic_confidences_mean": float(np.mean(synthetic_confidences)),
        "note": "A lower attack accuracy suggests the model is more robust against membership inference attacks."
    }
