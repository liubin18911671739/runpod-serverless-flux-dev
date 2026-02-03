FROM runpod/pytorch:2.1.2-py3.10-cuda12.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf \
    TRANSFORMERS_CACHE=/workspace/hf \
    DIFFUSERS_CACHE=/workspace/hf

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
