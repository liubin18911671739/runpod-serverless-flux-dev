import base64
import io
import os
import random
from typing import Any, Dict

import runpod
import torch
from diffusers import FluxPipeline
from PIL import Image

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
DTYPE = os.getenv("TORCH_DTYPE", "bfloat16").lower()
USE_CPU = os.getenv("USE_CPU", "false").lower() == "true"


def _get_dtype() -> torch.dtype:
    if DTYPE in ("bf16", "bfloat16"):
        return torch.bfloat16
    if DTYPE in ("fp16", "float16"):
        return torch.float16
    return torch.float32


DEVICE = "cpu" if USE_CPU or not torch.cuda.is_available() else "cuda"

# Load once at startup to reuse weights across requests.
pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=_get_dtype(),
)

if DEVICE == "cuda":
    pipe.to("cuda")
else:
    pipe.to("cpu")

# Conservative defaults for serverless usage.
pipe.set_progress_bar_config(disable=True)


def _encode_image(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_data = event.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "Missing required field: input.prompt"}

    negative_prompt = input_data.get("negative_prompt")
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))
    num_inference_steps = int(input_data.get("num_inference_steps", 30))
    guidance_scale = float(input_data.get("guidance_scale", 3.5))

    seed = input_data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image = result.images[0]
    return {
        "seed": seed,
        "image": _encode_image(image),
        "format": "png",
    }


runpod.serverless.start({"handler": handler})
