# Build stage
FROM quay.io/rose/rose-client

# Copy the source code into the container
COPY . /ml

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--driver", "/ml/driver.py"]
CMD ["--port", "8081"]

# --- Build Image ---
FROM registry.access.redhat.com/ubi9/python-39 AS build

WORKDIR /build

# Copy only the requirements file and install the Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# --- Runtime Image ---
FROM build

WORKDIR /app
COPY . /app

# Add the rose client package to the python path
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Default values for environment variables
ENV DRIVER ./mydriver.py
ENV PORT 8081

# Inform Docker that the container listens on port 3000
EXPOSE 8081

# Define the command to run your app using CMD which defines your runtime
CMD ["sh", "-c", "python rose/main.py --listen 0.0.0.0 --driver ${DRIVER} --port ${PORT}"]
