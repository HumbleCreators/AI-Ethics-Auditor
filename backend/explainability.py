import numpy as np
import joblib
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def generate_shap_explanation(content: bytes):
    """
    Generate an explanation using SHAP.
    Expects a pickled model file.
    Uses the Iris dataset for demonstration.
    """
    try:
        model = joblib.loads(content)
    except Exception as e:
        return {"error": f"Failed to load model for SHAP explanation: {str(e)}"}
    
    iris = load_iris()
    X_train, X_test, _, _ = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    
    try:
        explainer = shap.KernelExplainer(model.predict, X_train[:10])
        shap_values = explainer.shap_values(X_test[:1])
    except Exception as e:
        return {"error": f"Failed to generate SHAP explanation: {str(e)}"}
    
    explanation_summary = np.array(shap_values).mean(axis=1).tolist()
    return {
        "shap_summary": explanation_summary,
        "note": "SHAP explanation generated for one sample."
    }

def generate_lime_explanation(content: bytes):
    """
    Generate an explanation using LIME.
    Expects a pickled model file.
    """
    try:
        model = joblib.loads(content)
    except Exception as e:
        return {"error": f"Failed to load model for LIME explanation: {str(e)}"}
    
    lime_explanation = {
        "feature_importance": {"feature1": 0.5, "feature2": -0.3},
        "note": "Simulated LIME explanation for one sample."
    }
    return {"lime_explanation": lime_explanation}

def generate_counterfactual_explanation(content: bytes, input_sample: list):
    """
    Generate a counterfactual explanation by perturbing the input sample until the prediction flips.
    For simplicity, assumes a tabular model with numerical features.
    """
    try:
        model = joblib.loads(content)
    except Exception as e:
        return {"error": f"Failed to load model for counterfactual explanation: {str(e)}"}
    
    original_prediction = model.predict([input_sample])[0]
    
    step = 0.01
    counterfactual = input_sample.copy()
    
    max_iterations = 100
    iterations = 0
    while iterations < max_iterations:
        counterfactual = [x + step for x in counterfactual]
        new_prediction = model.predict([counterfactual])[0]
        if new_prediction != original_prediction:
            break
        iterations += 1
    
    if iterations == max_iterations:
        return {"note": "Counterfactual explanation not found within iteration limit.", "counterfactual": None}
    
    return {
        "original_prediction": original_prediction,
        "counterfactual": counterfactual,
        "iterations": iterations,
        "note": "Counterfactual explanation found by minimal perturbation."
    }
