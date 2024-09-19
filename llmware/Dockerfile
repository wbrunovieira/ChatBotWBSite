
FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt /app/


RUN pip install --upgrade pip && \
    pip uninstall -y tokenizers && \
    pip install tokenizers==0.13.3 && \
    pip install torch==2.0.1 -f https://download.pytorch.org/whl/cpu.html && \
    pip install transformers==4.31.0 &&\
    pip install -r requirements.txt
    

COPY . /app/


ENV PYTHONUNBUFFERED=1


EXPOSE 8501


CMD ["python", "main.py"]
