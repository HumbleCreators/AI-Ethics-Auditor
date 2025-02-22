import pandas as pd
from io import StringIO
import numpy as np
from .bias_detector import adversarial_debias

def mitigate_bias(content: bytes):
    """
    Mitigate bias in a dataset by applying reweighting and adversarial debiasing.
    Expects a CSV with a 'label' column.
    """
    try:
        data_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(data_str))
    except Exception as e:
        return {"error": f"Failed to parse CSV for mitigation: {str(e)}"}
    
    if 'label' not in df.columns:
        return {"error": "Dataset must include a 'label' column for mitigation."}
    
    # Simple reweighting based on label frequencies
    counts = df['label'].value_counts().to_dict()
    max_count = max(counts.values())
    df['sample_weight'] = df['label'].apply(lambda x: max_count / counts[x])
    
    # Applying adversarial debiasing simulation if a sensitive attribute is available
    if 'sensitive' in df.columns:
        df = adversarial_debias(df, 'sensitive', 'label')
    
    mitigated_csv = df.to_csv(index=False)
    return {"mitigated_dataset": mitigated_csv}
