FROM python:3.9-slim


RUN pip install -U pymilvus fastapi uvicorn spacy && \
python -m spacy download en_core_web_sm


WORKDIR /app

COPY app.py /app/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]