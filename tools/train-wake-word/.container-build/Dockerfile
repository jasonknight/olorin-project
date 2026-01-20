
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libsndfile1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch 2.x from official PyTorch index (required by piper-sample-generator)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Create constraints file to prevent torch/torchaudio from being changed
# Also pin numpy<2 for compatibility with pre-compiled packages
RUN echo "torch==2.1.0" > /tmp/constraints.txt && \
    echo "torchaudio==2.1.0" >> /tmp/constraints.txt && \
    echo "numpy<2" >> /tmp/constraints.txt

# Install numpy 1.x first
RUN pip install --no-cache-dir "numpy<2"

# Install openwakeword from source (train module not included in PyPI package)
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    git+https://github.com/dscripka/openWakeWord.git

# Install remaining training dependencies
# Using speechbrain>=1.0 which is compatible with torchaudio 2.x
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    "speechbrain>=1.0" \
    torchinfo \
    torchmetrics \
    tensorflow \
    audiomentations \
    torch-audiomentations \
    datasets \
    mutagen \
    pronouncing \
    deep-phonemizer \
    acoustics \
    pyyaml \
    onnx \
    tf2onnx

# piper-sample-generator v2.0.0 (has default model, compatible with openwakeword)
# Clone it so we have the generate_samples.py script available
RUN git clone --branch v2.0.0 --depth 1 https://github.com/rhasspy/piper-sample-generator.git /opt/piper-sample-generator && \
    cd /opt/piper-sample-generator && \
    pip install --no-cache-dir -c /tmp/constraints.txt -e . && \
    ls -la /opt/piper-sample-generator/

COPY train_in_container.py /app/
COPY config.yaml /app/

CMD ["python", "train_in_container.py"]
