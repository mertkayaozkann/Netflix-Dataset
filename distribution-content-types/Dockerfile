# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for curl, tar, and Spark
RUN apt-get update && apt-get install -y \
    curl \
    default-jre \
    tar \
    && apt-get clean

# Install Spark
RUN curl -sL https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz | tar -xz -C /opt \
    && ln -s /opt/spark-3.5.3-bin-hadoop3 /opt/spark

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Set the environment variable for Kaggle API key
ENV KAGGLE_CONFIG_DIR=/app/.kaggle

# Create a .kaggle directory in the container
RUN mkdir /app/.kaggle

# Copy the kaggle.json file into the .kaggle directory
COPY "kaggle(1).json" /app/.kaggle/kaggle.json

# Set the permissions for the kaggle.json file
RUN chmod 600 /app/.kaggle/kaggle.json

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the app
CMD ["python", "main.py"]
