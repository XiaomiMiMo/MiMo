# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and git
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# Clone the forked vLLM repository and install it
RUN git clone --branch feat_mimo_mtp_stable_073 https://github.com/XiaomiMiMo/vllm.git /opt/vllm && \
    pip3 install /opt/vllm

# Install other dependencies (e.g., transformers, torch)
# Note: vLLM setup.py should handle torch, but we specify a version compatible with CUDA 12.1
RUN pip3 install transformers==4.40.1 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Copy the MiMo registry script
COPY registry /app/registry

# Set the working directory
WORKDIR /app

# Expose the vLLM server port (default is 8000)
EXPOSE 8000

# Set up the entrypoint to run the vLLM server
# Users will need to specify the model path and other arguments
# Example: docker run -p 8000:8000 mimo-vllm --model /path/to/MiMo-7B-RL --trust-remote-code
# The model path will be mounted into the container by the user.
# We need to make sure the registry script is imported.
ENV PYTHONPATH="/app:$PYTHONPATH"

# Default command can be to show help, or a placeholder if model needs to be user-supplied
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--help"]
