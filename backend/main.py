from fastapi import FastAPI, UploadFile, File, HTTPException
from bias_detector import analyze_dataset_bias, analyze_model_bias
from explainability import generate_explanation
from mitigations import mitigate_bias
from db_manager import store_report

app = FastAPI(title="AI Ethics Auditor Backend")

@app.get("/")
def read_root():
    return {"message": "AI Ethics Auditor backend up and running."}

@app.post("/analyze/dataset")
async def analyze_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = analyze_dataset_bias(content)
        return {"bias_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/model")
async def analyze_model(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = analyze_model_bias(content)
        return {"model_bias": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
async def explain_prediction(file: UploadFile = File(...)):
    try:
        content = await file.read()
        explanation = generate_explanation(content)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mitigate")
async def mitigate(file: UploadFile = File(...)):
    try:
        content = await file.read()
        mitigated_data = mitigate_bias(content)
        store_report(mitigated_data)
        return {"mitigated_data": mitigated_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
