from fastapi import FastAPI, HTTPException
from llmware.models import ModelCatalog

app = FastAPI()


model_catalog = ModelCatalog()

@app.get("/")
def read_root():
    return {"message": "LLMWare Service running successfully"}


@app.get("/models")
def list_models():
    try:
        models = model_catalog.list_all_models()  
        return {"available_models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.post("/inference")
def run_inference(model_name: str, prompt: str):
    try:
        model = model_catalog.load_model(model_name)
        response = model.inference(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)

