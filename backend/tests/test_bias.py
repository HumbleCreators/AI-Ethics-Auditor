import io
import pandas as pd
import numpy as np
import joblib
import pytest
from backend.bias_detector import (
    analyze_dataset_bias,
    analyze_model_bias,
    compute_fairness_metrics,
    compute_intersectional_fairness
)

def test_analyze_dataset_bias_valid():
    # Create a simple CSV with a 'label' column.
    csv_data = "label\n0\n1\n0\n1\n1\n"
    content = csv_data.encode("utf-8")
    result = analyze_dataset_bias(content)
    # Verify that the analysis returns class counts and a fairness score.
    assert "class_counts" in result
    assert "fairness_score" in result
    assert isinstance(result["fairness_score"], float)

def test_analyze_dataset_bias_missing_label():
    # CSV missing 'label' column should return an error.
    csv_data = "feature\n1\n2\n3\n"
    content = csv_data.encode("utf-8")
    result = analyze_dataset_bias(content)
    assert "error" in result

def test_compute_fairness_metrics_valid():
    # Create a CSV with required columns: label, prediction, sensitive.
    csv_data = "label,prediction,sensitive\n0,0,0\n1,1,1\n0,0,0\n1,0,1\n"
    content = csv_data.encode("utf-8")
    result = compute_fairness_metrics(content)
    assert "group_accuracies" in result
    assert "overall_accuracy" in result

def test_compute_intersectional_fairness_valid():
    # Create a CSV with multiple sensitive attribute columns (prefixed with 'sensitive_').
    csv_data = "label,prediction,sensitive_gender,sensitive_age\n0,0,0,1\n1,1,1,1\n0,0,0,0\n1,1,1,0\n"
    content = csv_data.encode("utf-8")
    result = compute_intersectional_fairness(content)
    assert "intersectional_group_accuracies" in result
    assert "intersectional_fairness_gap" in result

def test_analyze_model_bias_valid():
    # Create a dummy model, fit it on Iris data, and pickle it.
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    content = joblib.dumps(model)
    result = analyze_model_bias(content)
    assert "confusion_matrix" in result
    assert "per_class_accuracy" in result
