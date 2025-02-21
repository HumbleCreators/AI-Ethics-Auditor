import numpy as np
import pandas as pd
from io import StringIO
import joblib
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def generate_explanation(content: bytes):
    """
    Generate an explanation for a model's prediction using SHAP.
    Expects a pickled model file.
    
    For demonstration, we use the Iris dataset. In production, you would pair the model with user-supplied data.
    """
    try:
        model = joblib.loads(content)
    except Exception as e:
        return {"error": f"Failed to load model for explanation: {str(e)}"}
    
    # Using Iris dataset for demonstration
    iris = load_iris()
    X_train, X_test, _, _ = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    
    # Use SHAP's KernelExplainer if the model is not tree-based.
    try:
        explainer = shap.KernelExplainer(model.predict, X_train[:10])
        shap_values = explainer.shap_values(X_test[:1])
    except Exception as e:
        return {"error": f"Failed to generate SHAP explanation: {str(e)}"}
    
    # For simplicity, we return a summary of SHAP values.
    explanation_summary = np.array(shap_values).mean(axis=1).tolist()
    return {"shap_summary": explanation_summary, "note": "SHAP explanation generated for one sample."}
