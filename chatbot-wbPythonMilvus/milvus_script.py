
from pymilvus import MilvusClient


client = MilvusClient("/var/lib/milvus/chatbot_vetores.db")


client.create_collection(
    collection_name="demo_collection",
    dimension=384 
)

print("Milvus Lite configurado com sucesso!")

while True:
    time.sleep(1000)