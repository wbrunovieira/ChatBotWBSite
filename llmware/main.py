from fastapi import FastAPI, HTTPException
from llmware.models import ModelCatalog

app = FastAPI()

# Initialize the ModelCatalog to explore available models
model_catalog = ModelCatalog()

@app.get("/")
def read_root():
    return {"message": "LLMWare Service running successfully"}

@app.post("/inference")
def run_inference(model_name: str, prompt: str):
    try:
        model = model_catalog.load_model(model_name)
        response = model.inference(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")
