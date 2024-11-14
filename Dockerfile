# Use a base image with Java and Python
FROM openjdk:11-jdk-slim

# Install Python, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y python3-pip wget unzip

# Download and install Apache SystemDS
RUN wget https://dlcdn.apache.org/systemds/3.2.0/systemds-3.2.0-bin.zip && \
    unzip systemds-3.2.0-bin.zip -d /opt/ && \
    rm systemds-3.2.0-bin.zip

# Set up the working directory
WORKDIR /app

# Copy your Python script and data files into the container
COPY linear_regression/systemds_lr.py /app/
COPY covid_features.csv /app/
COPY covid_target.csv /app/
COPY requirements.txt /app/

RUN pip3 install -r requirements.txt

# Run the Python script when the container starts
CMD ["python3", "systemds_lr.py"]
