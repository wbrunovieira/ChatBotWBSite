from fastapi import FastAPI
from pymilvus import MilvusClient

app = FastAPI()


client = MilvusClient("/var/lib/milvus/chatbot_vetores.db")


client.create_collection(
    collection_name="demo_collection",
    dimension=384 
)

@app.get("/")
def read_root():
    return {"message": "Milvus Lite configurado com sucesso!"}

@app.get("/create_collection/{name}")
def create_collection(name: str):
    client.create_collection(collection_name=name, dimension=384)
    return {"message": f"Coleção {name} criada com sucesso!"}

@app.get("/collections")
def list_collections():
    collections = client.list_collections()
    return {"collections": collections}
