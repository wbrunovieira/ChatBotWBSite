from fastapi import FastAPI, HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus

app = FastAPI()

# Initialize LangChain components
memory = ConversationBufferMemory()
vectorstore = Milvus(connection_args={"host": "milvus-lite", "port": "8000"})

# Example chain for conversational retrieval
retrieval_chain = ConversationalRetrievalChain(
    llm=OpenAIEmbeddings(),  # Replace with your specific LLM if needed
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
