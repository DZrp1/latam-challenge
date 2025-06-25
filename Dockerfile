# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for best practices
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model
COPY ./challenge ./challenge
COPY ./models ./models

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the app using the factory pattern
CMD exec uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}
