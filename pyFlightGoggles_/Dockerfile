FROM nvidia/cudagl:10.1-base-ubuntu18.04

# Set environment variables
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install required dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-venv \
    python3-virtualenv \
    libeigen3-dev \
    libopencv-dev \
    libzmqpp-dev \
    libblas-dev \
    ffmpeg \
    cmake \
    tmux \
    vim \
    nano \
    libffi-dev \
    libssl-dev \
    build-essential \
    libzmq3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Upgrade pip and install Python dependencies
RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -U virtualenv jupyter

# Set up Jupyter Notebook configuration
RUN mkdir -p /root/.jupyter && \
    jupyter notebook --generate-config && \
    echo 'c.NotebookApp.ip="0.0.0.0"' >> /root/.jupyter/jupyter_notebook_config.py && \
    echo 'c.NotebookApp.allow_root = True' >> /root/.jupyter/jupyter_notebook_config.py && \
    echo 'c.NotebookApp.token = ""' >> /root/.jupyter/jupyter_notebook_config.py

# Set working directory
WORKDIR /root/pyFlightGoggles

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook on container start
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
