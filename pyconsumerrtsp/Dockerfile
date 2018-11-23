# Start from a plain image with the latest version of python
FROM python:latest

# Author label
LABEL Author Adaickalavan Meiyappan

# Copy the local package files (from development computer) to the container's workspace (docker image)
COPY . /app 

# Set environment variables
ENV PYTHONPATH /app

# Install dependencies
RUN pip install -r /app/requirements.txt

# Command to execute when the image loads
ENTRYPOINT ["python", "/app/main.py"]
