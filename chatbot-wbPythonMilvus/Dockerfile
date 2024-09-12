FROM python:3.9-slim


RUN pip install -U pymilvus


WORKDIR /app

COPY milvus_script.py /app/milvus_script.py

CMD ["python", "milvus_script.py"]