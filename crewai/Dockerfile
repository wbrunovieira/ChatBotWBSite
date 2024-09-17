# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies specific to CrewAI
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port that the app will run on
EXPOSE 8502

# Run the application
CMD ["python", "main.py"]
