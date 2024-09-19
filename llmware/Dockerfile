
FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt /app/


RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY . /app/


ENV PYTHONUNBUFFERED=1


EXPOSE 8501


CMD ["python", "main.py"]
