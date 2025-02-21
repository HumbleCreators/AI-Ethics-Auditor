import pandas as pd
from io import StringIO
import numpy as np
from sklearn.metrics import confusion_matrix

def analyze_dataset_bias(content: bytes):
    """
    Analyze a CSV dataset for class imbalances.
    Expects a CSV file with a 'label' column.
    Returns class counts, imbalance ratios, and a fairness score.
    """
    try:
        data_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}"}
    
    if 'label' not in df.columns:
        return {"error": "Dataset must include a 'label' column."}
    
    counts = df['label'].value_counts().to_dict()
    max_count = max(counts.values())
    imbalance = {cls: max_count / count for cls, count in counts.items()}
    
    total = df.shape[0]
    proportions = np.array([count / total for count in counts.values()])
    fairness_score = 1 - np.std(proportions)  # Higher score means more balanced
    
    return {
        "class_counts": counts,
        "imbalance_ratios": imbalance,
        "fairness_score": fairness_score
    }

def analyze_model_bias(content: bytes):
    """
    Analyze a serialized model for bias.
    For demonstration, this function loads a pickled scikit-learn model,
    runs it on the Iris dataset, and computes a confusion matrix.
    """
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    try:
        model = joblib.loads(content)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions).tolist()
    
    per_class_accuracy = {}
    for i in range(len(cm)):
        true_positive = cm[i][i]
        total = sum(cm[i])
        per_class_accuracy[f"class_{i}"] = true_positive / total if total > 0 else 0
    
    return {
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy
    }

def compute_fairness_metrics(content: bytes):
    """
    Computes fairness metrics using fairlearn's MetricFrame.
    Expects a CSV with columns 'label', 'prediction', and 'sensitive'.
    """
    import pandas as pd
    from io import StringIO
    from fairlearn.metrics import MetricFrame, accuracy_score
    
    try:
        data_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}"}
    
    # Ensure required columns are present
    required = ['label', 'prediction', 'sensitive']
    if not all(col in df.columns for col in required):
        return {"error": f"CSV must include columns: {required}"}
    
    mf = MetricFrame(metrics=accuracy_score,
                     y_true=df['label'],
                     y_pred=df['prediction'],
                     sensitive_features=df['sensitive'])
    
    group_accuracies = mf.by_group.to_dict()
    overall_accuracy = mf.overall
    fairness_gap = max(group_accuracies.values()) - min(group_accuracies.values())
    
    return {
        "group_accuracies": group_accuracies,
        "overall_accuracy": overall_accuracy,
        "fairness_gap": fairness_gap
    }
