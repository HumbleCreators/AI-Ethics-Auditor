import pandas as pd
from io import StringIO
import numpy as np
from sklearn.metrics import confusion_matrix

def analyze_dataset_bias(content: bytes):
    """
    Analyze a CSV dataset for class imbalances.
    Expects a CSV file with a 'label' column.
    Returns class counts and imbalance ratios.
    """
    try:
        data_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}"}
    
    if 'label' not in df.columns:
        return {"error": "Dataset must include a 'label' column."}
    
    counts = df['label'].value_counts().to_dict()
    # Compute imbalance ratio: ratio between the maximum and each class count.
    max_count = max(counts.values())
    imbalance = {cls: max_count / count for cls, count in counts.items()}
    
    # Optionally, add a simple fairness metric (variance of the class proportions)
    total = df.shape[0]
    proportions = np.array([count / total for count in counts.values()])
    fairness_score = 1 - np.std(proportions)  # Higher means more balanced
    
    return {
        "class_counts": counts,
        "imbalance_ratios": imbalance,
        "fairness_score": fairness_score
    }

def analyze_model_bias(content: bytes):
    """
    Analyze a serialized model for bias.
    For demonstration, this function loads a pickled scikit-learn model,
    runs it on a fixed sample dataset (embedded here), and computes a confusion matrix.
    
    In a production system, you would allow the user to supply a model
    and associated test data.
    """
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    try:
        # Load the model from the uploaded file (expecting a pickle file)
        model = joblib.loads(content)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    # For demonstration, we use Iris dataset as test data.
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions).tolist()
    
    # A dummy fairness metric: ratio of correct predictions per class.
    per_class_accuracy = {}
    for i in range(cm.__len__()):
        true_positive = cm[i][i]
        total = sum(cm[i])
        per_class_accuracy[f"class_{i}"] = true_positive / total if total > 0 else 0
    
    return {
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy
    }
