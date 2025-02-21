import pandas as pd
from io import StringIO
import numpy as np

def mitigate_bias(content: bytes):
    """
    Mitigate bias in a dataset by applying a simple reweighting scheme.
    Expects a CSV with a 'label' column. For each class, the sample weight is
    adjusted inversely proportional to its frequency.
    """
    try:
        data_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse CSV for mitigation: {str(e)}"}
    
    if 'label' not in df.columns:
        return {"error": "Dataset must include a 'label' column for mitigation."}
    
    # Calculate class frequencies
    counts = df['label'].value_counts().to_dict()
    max_count = max(counts.values())
    df['sample_weight'] = df['label'].apply(lambda x: max_count / counts[x])
    
    # Optionally, you might want to resample or adjust weights during training.
    # Here we simply return the dataset with the added sample_weight column.
    mitigated_csv = df.to_csv(index=False)
    return {"mitigated_dataset": mitigated_csv}
