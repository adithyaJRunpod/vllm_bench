FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

ARG VLLM_VERSION=0.9.1
ARG CUDA_VERSION=cu121
ARG TORCH_VERSION=torch2.3

RUN python3 -m pip install vllm==${VLLM_VERSION} && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/${CUDA_VERSION}/${TORCH_VERSION}

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src

WORKDIR /src

CMD ["python3", "handler.py"]
