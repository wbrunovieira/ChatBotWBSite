from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import MilvusClient
import spacy
import numpy as np

app = FastAPI()


client = MilvusClient("/var/lib/milvus/chatbot_vetores.db")


nlp = spacy.load("en_core_web_sm")

VECTOR_DIMENSION = 96


class TextData(BaseModel):
    text: str
    collection_name: str


@app.get("/create_collection/{name}")
def create_collection(name: str):
    try:
        client.create_collection(
            collection_name=name,
            dimension=VECTOR_DIMENSION
        )
        return {"message": f"Coleção {name} criada com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao criar coleção: {str(e)}")


@app.post("/add_text/")
def add_text(data: TextData):
    try:
        doc = nlp(data.text)
        embedding = doc.vector

   
        embedding = np.array(embedding, dtype=np.float32).tolist()

        if len(embedding) != VECTOR_DIMENSION:
            raise HTTPException(status_code=400, detail=f"Dimensão do vetor incorreta: {len(embedding)}. Esperado: {VECTOR_DIMENSION}.")

        client.insert(
            collection_name=data.collection_name,
            data=[embedding]
        )
        return {"message": "Texto inserido com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao inserir dados: {str(e)}")


@app.get("/collections")
def list_collections():
    try:
        collections = client.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar coleções: {str(e)}")


@app.get("/collection_schema/{name}")
def collection_schema(name: str):
    try:
        collection = client.get_collection_schema(name)
        return {"schema": collection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter o esquema da coleção: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Milvus Lite configurado com sucesso!"}
