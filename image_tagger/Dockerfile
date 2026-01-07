FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=All

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python-Pakete
RUN pip3 install --no-cache-dir \
    torch torchvision \
    facenet-pytorch \
    pillow \
    tqdm

# Arbeitsverzeichnis
WORKDIR /app

#COPY label_faces.py .

#CMD ["python3", "label_faces.py"]
