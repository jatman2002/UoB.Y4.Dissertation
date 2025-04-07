FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install required Python packages
RUN pip3 install keras==3.6.0 numpy==1.24.3 pandas==2.0.3

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh