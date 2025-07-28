import torch
from PIL import Image
from .diffusers.utils import load_image
from .flux_controlnet import FluxControlNetModel
from .flux_controlnet import FluxTransformer2DModel
from .flux_controlnet import FluxControlNetInpaintingPipeline

# Charge les modèles une seule fois
print("[Flux-ControlNet] Loading models...")
controlnet = FluxControlNetModel.from_pretrained(
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    torch_dtype=torch.bfloat16
)
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

def inpaint(image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
    """
    Inpainting avec FLUX-ControlNet

    Args:
        image: PIL image RGB
        mask: PIL image RGB (zone à remplacer = blanc) 
        
        prompt: prompt textuel

    Returns:
        Image inpaintée (PIL.Image)
    """
    # Resize standard
    size = (768, 768)
    image = image.convert("RGB").resize(size)
    mask = mask.convert("RGB").resize(size)

    # Déterminisme
    generator = torch.Generator(device="cuda").manual_seed(24)

    # Appel du pipeline
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=1.0
    ).images[0]

    return result
