import os
import time
import json
from scripts import model_sd15, model_sdxl, model_lama
from scripts.evaluate import evaluate_image
from scripts.utils import load_image, load_mask, save_result

MODELS = {
    "sd15": model_sd15.inpaint,
    "sdxl": model_sdxl.inpaint,
    "lama": model_lama.inpaint
}

with open("prompts/prompts.json") as f:
    prompts = json.load(f)

for model_name, model_fn in MODELS.items():
    os.makedirs(f"data/results/{model_name}", exist_ok=True)
    for fname in os.listdir("data/images"):
        image = load_image(f"data/images/{fname}")
        mask = load_mask(f"data/masks/{fname}")
        prompt = prompts[fname]

        start = time.time()
        result = model_fn(image, mask, prompt)
        duration = time.time() - start

        save_result(result, f"data/results/{model_name}/{fname}")
        evaluate_image(fname, model_name, result, duration)
