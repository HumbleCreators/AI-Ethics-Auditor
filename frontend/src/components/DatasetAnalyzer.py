import pandas as pd
from io import StringIO

def validate_dataset(file_bytes: bytes):
    """
    Validates the uploaded CSV file.
    Returns a tuple: (is_valid: bool, message: str, dataframe: pd.DataFrame or None)
    """
    try:
        decoded = file_bytes.decode("utf-8")
        df = pd.read_csv(StringIO(decoded))
        if 'label' not in df.columns:
            return (False, "CSV must include a 'label' column.", None)
        return (True, "Dataset is valid.", df)
    except Exception as e:
        return (False, f"Error reading CSV: {e}", None)

def get_dataset_summary(df: pd.DataFrame):
    """
    Returns summary statistics of the dataset.
    """
    summary = {
        "num_rows": df.shape[0],
        "columns": list(df.columns),
        "head": df.head(3).to_dict(orient='records')
    }
    return summary
