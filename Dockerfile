# GLOBAL BUILD LIBRARIES
# Use NVIDIA Triton Inference server for GPU acceleration
FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 as base

# pull NVIDIA container toolkit repo
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# install dependencies
RUN apt-get update
RUN apt-get install python-is-python3
RUN apt-get install -y nvidia-container-toolkit
RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# set CWD
WORKDIR /usr/local/bin

# LOCAL BUILD LIBRARIES
# Split - Python installs
FROM base as dev

# Copy local files
COPY . .

# install dependencies
RUN pip3 install -r requirements.txt