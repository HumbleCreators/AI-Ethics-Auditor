import io
import pandas as pd
import pytest
from backend.mitigations import mitigate_bias

def test_mitigate_bias_valid():
    # Create a CSV with a 'label' column and some dummy features.
    csv_data = "label,feature,sensitive\n0,1.0,0\n1,2.0,1\n0,1.5,0\n1,3.0,1\n"
    content = csv_data.encode("utf-8")
    result = mitigate_bias(content)
    # Check that the mitigated output includes the expected key.
    assert "mitigated_dataset" in result
    # Verify that the returned CSV now has a 'sample_weight' column.
    df = pd.read_csv(io.StringIO(result["mitigated_dataset"]))
    assert "sample_weight" in df.columns
    # Also check if 'adversarial_weight' column exists since sensitive attribute is provided.
    assert "adversarial_weight" in df.columns

def test_mitigate_bias_missing_label():
    # CSV without a 'label' column should return an error.
    csv_data = "feature,sensitive\n1.0,0\n2.0,1\n1.5,0\n3.0,1\n"
    content = csv_data.encode("utf-8")
    result = mitigate_bias(content)
    assert "error" in result
