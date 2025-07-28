import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

def load():
    print("[FLUX-Kontext] Loading model...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    return pipe

def inpaint(image, mask, prompt, pipe):
    """
    Inpainting implicite avec FLUX-Kontext (pas de masque)

    Args:
        image: PIL image RGB
        prompt: prompt textuel

    Returns:
        Image modifiée (PIL.Image)
    """
    size = (512, 512)
    image = image.convert("RGB").resize(size)

    result = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=2.5
    ).images[0]

    return result, None

if __name__ == "__main__":
    image_path = "/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/bed.png"
    prompt = "add a dog lying on the bed"

    image = load_image(image_path)
    pipe = load()
    result_image = inpaint(image, prompt, pipe)
    result_image.save("/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/flux_kontext_result.png")
    print("Inpainting completed and saved as 'flux_kontext_result.png'")
