from fastapi import FastAPI, UploadFile, File, HTTPException
from bias_detector import (
    analyze_dataset_bias,
    analyze_model_bias,
    compute_fairness_metrics
)
from explainability import generate_shap_explanation, generate_lime_explanation
from mitigations import mitigate_bias
from privacy import perform_privacy_tests
from db_manager import store_report

app = FastAPI(title="AI Ethics Auditor Backend", version="1.1")

@app.get("/")
def read_root():
    return {"message": "AI Ethics Auditor backend up and running."}

@app.post("/analyze/dataset")
async def endpoint_analyze_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = analyze_dataset_bias(content)
        return {"bias_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/model")
async def endpoint_analyze_model(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = analyze_model_bias(content)
        return {"model_bias": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/fairness")
async def endpoint_analyze_fairness(file: UploadFile = File(...)):
    """
    Expects a CSV with columns: 'label', 'prediction', 'sensitive'.
    """
    try:
        content = await file.read()
        result = compute_fairness_metrics(content)
        return {"fairness_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/privacy")
async def endpoint_analyze_privacy(file: UploadFile = File(...)):
    """
    Expects a pickled model file.
    """
    try:
        content = await file.read()
        result = perform_privacy_tests(content)
        return {"privacy_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/shap")
async def endpoint_explain_shap(file: UploadFile = File(...)):
    """
    Expects a pickled model file for SHAP explanation.
    """
    try:
        content = await file.read()
        explanation = generate_shap_explanation(content)
        return {"shap_explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/lime")
async def endpoint_explain_lime(file: UploadFile = File(...)):
    """
    Expects a pickled model file for LIME explanation.
    """
    try:
        content = await file.read()
        explanation = generate_lime_explanation(content)
        return {"lime_explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mitigate")
async def endpoint_mitigate(file: UploadFile = File(...)):
    """
    Expects a CSV with a 'label' column.
    Applies a simple reweighting scheme to mitigate bias.
    """
    try:
        content = await file.read()
        mitigated_data = mitigate_bias(content)
        store_report(mitigated_data)
        return {"mitigated_data": mitigated_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
