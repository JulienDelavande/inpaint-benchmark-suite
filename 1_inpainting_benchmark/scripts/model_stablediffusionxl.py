import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

def load():
    print("[SDXL-Inpaint] Loading model...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def inpaint(image, mask, prompt, pipe):
    """
    Inpainting avec Stable Diffusion XL 1.0 Inpainting

    Args:
        image: PIL image RGB
        mask: PIL image RGB (zone à remplacer = blanc)
        prompt: prompt textuel

    Returns:
        Image inpaintée (PIL.Image)
    """
    size = (1024, 1024)
    image = image.convert("RGB").resize(size)
    mask = mask.convert("RGB").resize(size)

    generator = torch.Generator(device="cuda").manual_seed(92)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        generator=generator,
    ).images[0]

    return result, mask

if __name__ == "__main__":
    image_path = "/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/bed.png"
    mask_path = "/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/bed_mask.png"
    prompt = "add a dog lying on the bed"

    image = load_image(image_path)
    mask = load_image(mask_path)

    pipe = load()
    result_image, _ = inpaint(image, mask, prompt, pipe)
    result_image.save("/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/sdxl_inpaint_result.png")
    print("Inpainting completed and saved as 'sdxl_inpaint_result.png'")
