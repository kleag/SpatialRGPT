# --- Stage 1: Build stage ---
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS builder

# Install Python 3.12 and build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    aria2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
     && rm -rf /var/lib/apt/lists/*

# Assuming your cert is named 'cea-ca.crt' in your local folder
COPY cea-ca.crt /usr/local/share/ca-certificates/cea-ca.crt

# Install ca-certificates and update the store
RUN apt-get update && apt-get install -y ca-certificates && \
    update-ca-certificates

# Install uv from the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# UV Configuration for strictness and performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_NATIVE_TLS=1
ENV UV_PYTHON=python3.12

# Copy lockfiles first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a localized .venv
# --frozen: ensures the lockfile is respected exactly
# --no-install-project: avoids installing the local package in the build layer
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --no-dev --native-tls


# 2. Clone External Repositories (Pinning to your specific commit)
RUN mkdir -p external && cd external && \
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git && \
    cd Grounded-Segment-Anything && \
    git checkout 126abe633ffe333e16e4a0a4e946bc1003caf757

RUN cd external/Grounded-Segment-Anything/segment_anything && \
    echo "hello" && \
    uv pip install -e .

# 3. Install Grounded-SAM components
# We use 'uv pip install' within the managed environment
RUN cd external/Grounded-Segment-Anything && \
#    uv pip install -e segment_anything && \
    # GroundingDINO requires --no-build-isolation to use the environment's torch/cuda
    uv pip install --no-build-isolation -e GroundingDINO

# 4. Install RAM (Recognize Anything)
RUN cd external/Grounded-Segment-Anything && \
    git clone https://github.com/xinyu1205/recognize-anything.git && \
    uv pip install -r ./recognize-anything/requirements.txt && \
    uv pip install -e ./recognize-anything/

RUN cd external/Grounded-Segment-Anything/segment_anything && \
    echo "hello" && \
    uv pip install -e .

# Necessary as one of the external install upgrades numpy, which
# is not compatible with other packages.

RUN uv pip install numpy==1.26.0
# # 5. Install Perspective Fields
# RUN cd external && \
#     git clone https://github.com/jinlinyi/PerspectiveFields.git && \
#     cd PerspectiveFields && \
#     # We install the dependencies first, then the package itself
#     uv pip install -r requirements.txt && \
#     uv pip install -e .

# # Set Environment Variable for PerspectiveFields (if required by your code)
# ENV PERSPECTIVE_FIELDS_PATH="/app/external/PerspectiveFields"


# --- Stage 2: Runtime stage ---
#FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

# Install Python 3.12 runtime and Computer Vision libraries
# libgl1 and libglib2.0-0 are required for OpenCV/Detectron2
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    software-properties-common \
#    && add-apt-repository ppa:deadsnakes/ppa \
#    && apt-get update && apt-get install -y --no-install-recommends \
#    python3.12 \
#    libgl1 \
#    libglib2.0-0 \
#    libgomp1 \
#    aria2 \
#    && rm -rf /var/lib/apt/lists/*

#WORKDIR /app

# Copy the virtual environment from the builder stage
#COPY --from=builder /app/.venv /app/.venv

# Copy your source code
# Copy your application source code
COPY dataset_pipeline/osdsynth/processor osdsynth/processor
COPY dataset_pipeline/osdsynth/utils osdsynth/utils
COPY dataset_pipeline/osdsynth/visualizer osdsynth/visualizer
COPY dataset_pipeline/configs configs
COPY dataset_pipeline/demo_images demo_images
COPY dataset_pipeline/scripts scripts
COPY dataset_pipeline/__init__.py .
COPY dataset_pipeline/run_scene_graph.py .
COPY dataset_pipeline/app.py .

# Create a directory for weights
RUN mkdir -p /app/weights

# Copy a helper script to manage the download
COPY dataset_pipeline/scripts/download_all_weights_docker.sh /app/scripts/download_all_weights.sh
RUN chmod +x /app/scripts/download_all_weights.sh

ENV DEPTH_ANYTHING_PATH=/app/weights/depth_anything
ENV SAM_CKPT_PATH=/app/weights/grounded_sam/sam_hq_vit_h.pth
ENV RAM_CKPT_PATH=/app/weights/grounded_sam/ram_swin_large_14m.pth
ENV CKPT=/app/weights/SpatialRGPT-VILA1.5-8B/
ENV GSA_PATH=/app/weights/grounded_sam

# Environment setup
ENV PATH="/app/.venv/bin:$PATH"
# Ensure the dynamic linker can find CUDA 13.0 libraries
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Expose FastAPI's default port
EXPOSE 8000

# Use a shell script to check weights and start FastAPI
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


