# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure logs are output in real-time
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and app code
COPY model ./model
COPY app ./app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]



## OLD - Use the official Python image from the Docker Hub
#FROM python:3.9-slim
#
## Set the working directory
#WORKDIR /app
#
## Copy the requirements file and install dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy the model and app code
#COPY model ./model
#COPY app ./app
#
## Expose the port the app runs on
#EXPOSE 8000
#
## Command to run the application
#CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
