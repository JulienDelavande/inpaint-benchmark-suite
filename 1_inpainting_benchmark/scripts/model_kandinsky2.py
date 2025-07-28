import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

def load():
    print("[Kandinsky-Inpaint] Loading model...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def inpaint(image, mask, prompt, pipe):
    """
    Inpainting avec Kandinsky 2.2 Decoder Inpainting

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
    result_image.save("/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/kandinsky_inpaint_result.png")
    print("Inpainting completed and saved as 'kandinsky_inpaint_result.png'")
