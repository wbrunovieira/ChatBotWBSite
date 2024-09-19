from fastapi import FastAPI, HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_milvus import MilvusVectorStore

app = FastAPI()


memory = ConversationBufferMemory()

vectorstore = MilvusVectorStore(
    connection_args={"host": "milvus-lite", "port": "8000"},
    embedding_function=OpenAIEmbeddings()
)


retrieval_chain = ConversationalRetrievalChain(
    llm=OpenAIEmbeddings(),  
    retriever=vectorstore.as_retriever(),
    memory=memory
)

@app.get("/")
def read_root():
    return {"message": "LangChain Service running successfully"}

@app.post("/query")
def query(prompt: str):
    try:
        response = retrieval_chain.run(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")
