import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

def load():
    print("[SD2-Inpaint] Loading model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")
    return pipe

def inpaint(image, mask, prompt, pipe):
    """
    Inpainting avec Stable Diffusion 2 Inpainting

    Args:
        image: PIL image RGB
        mask: PIL image RGB (zone à remplacer = blanc)
        prompt: prompt textuel

    Returns:
        Image inpaintée (PIL.Image)
    """
    size = (512, 512)
    image = image.convert("RGB").resize(size)
    mask = mask.convert("RGB").resize(size)

    generator = torch.Generator(device="cuda").manual_seed(24)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        generator=generator,
        # num_inference_steps=25,
        # guidance_scale=7.5,
    ).images[0]

    return result, mask

if __name__ == "__main__":
    image_path = "/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/bed.png"
    mask_path = "/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/bed_mask.png"
    prompt = "add a dog lying on the bed"

    image = load_image(image_path)
    mask = load_image(mask_path)

    result_image, _ = inpaint(image, mask, prompt)
    result_image.save("/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/sd2_inpaint_result.png")
    print("Inpainting completed and saved as 'sd2_inpaint_result.png'")
