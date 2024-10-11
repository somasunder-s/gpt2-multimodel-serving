# Use the official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on (default for FastAPI is 8000)
EXPOSE 8000

# Command to run the application with Gunicorn and Uvicorn workers
CMD ["gunicorn", "app.app:app", "-w", "10", "-k", "uvicorn.workers.UvicornWorker", "--threads", "32", "--bind", "0.0.0.0:8000"]
