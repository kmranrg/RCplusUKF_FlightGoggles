FROM nvidia/cudagl:10.1-base-ubuntu18.04

ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key list

# Install dependencies and Python 3.8
RUN apt-get update && apt-get install -y \
  software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt-get update && apt-get install -y \
  python3.8 \
  python3.8-dev \
  python3.8-venv \
  python3.8-distutils \
  libeigen3-dev \
  libopencv-dev \
  libzmqpp-dev \
  libblas-dev \
  libffi-dev \
  libssl-dev \
  build-essential \
  ffmpeg \
  cmake \
  tmux \
  vim \
  nano \
  wget \
  && apt-get clean

# Manually install pip for Python 3.8
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.8 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

RUN python3.8 -m pip install -U virtualenv jupyter

ENTRYPOINT ["/bin/bash", "-c", "jupyter notebook --generate-config && \
    echo 'c.NotebookApp.ip=\"127.0.0.1\"' >> /root/.jupyter/jupyter_notebook_config.py && \
    echo 'c.NotebookApp.allow_root = True' >> /root/.jupyter/jupyter_notebook_config.py && \
    cd /root/pyFlightGoggles && \
    exec /bin/bash"]
