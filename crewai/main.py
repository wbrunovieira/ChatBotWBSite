from fastapi import FastAPI, HTTPException


__all__ = ["CrewManager"]

app = FastAPI()


class CrewManager:
    def orchestrate(self, task: str, agents: list):
      
        return f"Orchestrating {task} with {len(agents)} agents"


crew_manager = CrewManager()


@app.get("/")
def read_root():
    return {"message": "CrewAI Service running successfully"}


@app.post("/orchestrate-task")
def orchestrate_task(task: str, agents: list):
    try:
        
        result = crew_manager.orchestrate(task, agents)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error orchestrating task: {str(e)}")
