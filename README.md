# RunPod Serverless Flux Dev Demo

This repo is a minimal RunPod Serverless worker that runs a Flux image generation pipeline with Hugging Face Diffusers.

## Files
- `handler.py`: serverless handler
- `requirements.txt`: Python deps
- `Dockerfile`: container image for RunPod Serverless
- `sample_request.json`: example request payload

## Quick start
1. Build and push the image to your registry.
2. Create a RunPod Serverless endpoint using that image.
3. Send a request using the payload in `sample_request.json`.

## Environment variables
- `MODEL_ID`: Hugging Face model id (default: `black-forest-labs/FLUX.1-dev`).
- `HUGGINGFACE_HUB_TOKEN`: required if the model is gated.
- `TORCH_DTYPE`: `bfloat16`, `float16`, or `float32` (default: `bfloat16`).
- `USE_CPU`: set `true` to force CPU (for debugging only).

## Notes
- Flux models are large and expect a GPU. Allocate enough VRAM for the requested width/height.
- Defaults are conservative for serverless inference. Adjust `num_inference_steps` and `guidance_scale` per your use case.
- The Docker base image already includes a CUDA-matched PyTorch build, so `torch` is not pinned in `requirements.txt`.

# runpod-serverless-flux-dev
# runpod-serverless-flux-dev
