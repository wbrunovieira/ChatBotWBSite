from fastapi import FastAPI, HTTPException
from copilotkit import Copilot

app = FastAPI()


copilot = Copilot()

@app.get("/")
def read_root():
    return {"message": "CopilotKit Service running successfully"}

@app.post("/perform-action")
def perform_action(action: str, state: dict):
    try:
        result = copilot.perform_action(action, state)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing action: {str(e)}")
