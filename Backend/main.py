import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from .bias_detector import (
    analyze_dataset_bias,
    analyze_model_bias,
    compute_fairness_metrics
)
from .explainability import generate_shap_explanation, generate_lime_explanation
from .mitigations import mitigate_bias
from .privacy import perform_privacy_tests
from .db_manager import store_report
import pandas as pd
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Ethics Auditor Backend", version="1.3")

# SECURITY: Simple API key dependency for demonstration purposes.
API_KEY = "secret-token"

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        logger.warning("Unauthorized access attempt with API key: %s", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# Utility function for consistent JSON responses.
def consistent_response(success: bool, data=None, error=None):
    return {"success": success, "data": data, "error": error}

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return consistent_response(True, data={"message": "AI Ethics Auditor backend up and running."})

@app.post("/analyze/dataset", dependencies=[Depends(verify_api_key)])
async def endpoint_analyze_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # Input validation: check for non-empty content.
        if not content:
            raise ValueError("Uploaded file is empty.")
        # Validate CSV contains a 'label' column
        df = pd.read_csv(StringIO(content.decode("utf-8")))
        if 'label' not in df.columns:
            raise ValueError("CSV file must include a 'label' column.")
        result = analyze_dataset_bias(content)
        logger.info("Dataset analysis completed successfully.")
        return consistent_response(True, data={"bias_analysis": result})
    except Exception as e:
        logger.error("Error in /analyze/dataset: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error=str(e)))

@app.post("/analyze/model", dependencies=[Depends(verify_api_key)])
async def endpoint_analyze_model(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded model file is empty.")
        result = analyze_model_bias(content)
        logger.info("Model analysis completed successfully.")
        return consistent_response(True, data={"model_bias": result})
    except Exception as e:
        logger.error("Error in /analyze/model: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error=str(e)))

@app.post("/analyze/fairness", dependencies=[Depends(verify_api_key)])
async def endpoint_analyze_fairness(file: UploadFile = File(...)):
    """
    Expects a CSV with columns: 'label', 'prediction', 'sensitive'.
    """
    try:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded CSV file is empty.")
        # Validate required columns
        df = pd.read_csv(StringIO(content.decode("utf-8")))
        for col in ['label', 'prediction', 'sensitive']:
            if col not in df.columns:
                raise ValueError(f"CSV file must include a '{col}' column.")
        result = compute_fairness_metrics(content)
        logger.info("Fairness analysis completed successfully.")
        return consistent_response(True, data={"fairness_analysis": result})
    except Exception as e:
        logger.error("Error in /analyze/fairness: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error=str(e)))

# Updated privacy endpoint now accepts two files: model and training dataset.
@app.post("/analyze/privacy", dependencies=[Depends(verify_api_key)])
async def endpoint_analyze_privacy(
    model: UploadFile = File(...), 
    train: UploadFile = File(...)
):
    """
    Expects:
      - 'model': a pickled model file.
      - 'train': a CSV file containing the training data.
    """
    try:
        model_content = await model.read()
        train_content = await train.read()
        if not model_content or not train_content:
            raise ValueError("Both model and training files must be provided and non-empty.")
        result = perform_privacy_tests(model_content, train_content)
        logger.info("Privacy analysis completed successfully.")
        return consistent_response(True, data={"privacy_analysis": result})
    except Exception as e:
        logger.error("Error in /analyze/privacy: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error="An internal error has occurred. Please try again later."))

@app.post("/explain/shap", dependencies=[Depends(verify_api_key)])
async def endpoint_explain_shap(file: UploadFile = File(...)):
    """
    Expects a pickled model file for SHAP explanation.
    """
    try:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded model file is empty.")
        explanation = generate_shap_explanation(content)
        logger.info("SHAP explanation generated successfully.")
        return consistent_response(True, data={"shap_explanation": explanation})
    except Exception as e:
        logger.error("Error in /explain/shap: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error="An internal error has occurred. Please try again later."))

@app.post("/explain/lime", dependencies=[Depends(verify_api_key)])
async def endpoint_explain_lime(file: UploadFile = File(...)):
    """
    Expects a pickled model file for LIME explanation.
    """
    try:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded model file is empty.")
        explanation = generate_lime_explanation(content)
        logger.info("LIME explanation generated successfully.")
        return consistent_response(True, data={"lime_explanation": explanation})
    except RuntimeError as e:
        logger.error("Error in /explain/lime: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error="An internal error has occurred. Please try again later."))
    except Exception as e:
        logger.error("Unexpected error in /explain/lime: %s", str(e))
        raise HTTPException(status_code=500, detail=consistent_response(False, error="An unexpected error has occurred. Please try again later."))

@app.post("/mitigate", dependencies=[Depends(verify_api_key)])
async def endpoint_mitigate(file: UploadFile = File(...)):
    """
    Expects a CSV with a 'label' column.
    Applies a simple reweighting scheme to mitigate bias.
    """
    try:
        content = await file.read()
        if not content:
            raise ValueError("Uploaded CSV file is empty.")
        df = pd.read_csv(StringIO(content.decode("utf-8")))
        if 'label' not in df.columns:
            raise ValueError("CSV file must include a 'label' column.")
        mitigated_data = mitigate_bias(content)
        store_report(mitigated_data)
        logger.info("Bias mitigation completed and report stored.")
        return consistent_response(True, data={"mitigated_data": mitigated_data})
    except Exception as e:
        logger.error("Error in /mitigate: %s", str(e))
        raise HTTPException(status_code=400, detail=consistent_response(False, error="An internal error has occurred. Please try again later."))