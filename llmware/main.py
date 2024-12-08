from fastapi import FastAPI, HTTPException
from llmware.models import ModelCatalog

app = FastAPI()


model_catalog = ModelCatalog()

embedding_model_name = "all-MiniLM-L6-v2"
conversation_model_name = "bling-phi-3.5-gguf"

embedding_model = model_catalog.load_model(embedding_model_name)
conversation_model = model_catalog.load_model(conversation_model_name)

@app.post("/embedding")
def generate_embedding(text: str):
    try:
        embedding = embedding_model.inference(text)  
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.post("/conversation")
def generate_conversation(prompt: str):
    try:
        response = conversation_model.inference(prompt)  
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating conversation: {str(e)}")
        
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

