import os
import csv
import json
import argparse
import torch
from PIL import Image
#import model_fluxcontrolbeta
import model_stablediffusionv2
import model_kandinsky2
import model_stablediffusion
import model_stablediffusionxl
import model_fluxkontextdev

# Dictionnaire de modèles
MODELS_INPAINT = {
    #"fluxcontrolbeta": model_fluxcontrolbeta.inpaint,
    "stablediffusionv2": model_stablediffusionv2.inpaint,
    "kandinsky": model_kandinsky2.inpaint,
    "stablediffusion": model_stablediffusion.inpaint,
    "stablediffusionxl": model_stablediffusionxl.inpaint,
    "fluxkontextdev": model_fluxkontextdev.inpaint
}

# Paramètres modèles à logguer
MODEL_CONFIGS = {
    #"fluxcontrolbeta": {"version": "beta", "source": "FluxControlNet"},
    "stablediffusionv2" : {"version": "2.0", "source": "StableDiffusionInpaint"},
    "kandinsky": {"version": "2.2", "source": "Kandinsky-2.2"},
    "stablediffusion": {"version": "1.5", "source": "StableDiffusionInpaint"},
    "stablediffusionxl": {"version": "1.0", "source": "StableDiffusionXL"},
    "fluxkontextdev": {"version": "1.0", "source": "FLUX-Kontext-dev"}
}

MODELS_LOAD = {
    #"fluxcontrolbeta": model_fluxcontrolbeta.load,
    "stablediffusionv2": model_stablediffusionv2.load,
    "kandinsky": model_kandinsky2.load,
    "stablediffusion": model_stablediffusion.load,
    "stablediffusionxl": model_stablediffusionxl.load,
    "fluxkontextdev": model_fluxkontextdev.load
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_csv(csv_path, row, header=None):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header or row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def load_local_dataset(csv_file, data_dir):
    dataset = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {
                "image_path": os.path.join(data_dir, row["image_path"]),
                "mask_path": os.path.join(data_dir, row["mask_path"]),
                "prompt": row["prompt"]
            }
            dataset.append(sample)
    return dataset

def benchmark_dataset(model_name, inpaint_fn, load_fn, dataset, result_dir, csv_path, num_samples=1):
    ensure_dir(result_dir)
    header = ["model_name", "image", "prompt", "image_generated", "duration_sec", "memory_MB", "model_params"]

    pipe = load_fn()  # Load the model

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("RGB")
        prompt = sample["prompt"]
        image_name = os.path.basename(sample["image_path"]).split('.')[0]
        fname = f"{image_name}_result.png"

        mask = mask.convert("L")  # convert to grayscale
        threshold = 253
        mask = mask.point(lambda p: 255 if p > threshold else 0).convert("RGB")


        # CUDA timing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        starter.record()

        result, mask = inpaint_fn(image, mask, prompt, pipe)

        ender.record()
        torch.cuda.synchronize()

        duration_ms = starter.elapsed_time(ender)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Save result image
        output_path = os.path.join(result_dir, fname)
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu()
            result = result.permute(1, 2, 0).numpy() * 255
            result = Image.fromarray(result.astype("uint8"))
        result.save(output_path)
        if mask is not None:
            mask.save(os.path.join(result_dir, f"mask_{image_name}.png"))

        # Log CSV
        row = {
            "model_name": model_name,
            "image": sample["image_path"],
            "prompt": prompt,
            "image_generated": output_path,
            "duration_sec": round(duration_ms / 1000, 3),
            "memory_MB": round(peak_mem, 2),
            "model_params": json.dumps(MODEL_CONFIGS[model_name])
        }
        log_csv(csv_path, row, header)

        print(f"[{model_name}] {fname} | {duration_ms:.2f} ms | {peak_mem:.2f} MB")

def main():
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, default="/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/inpainting_prompts.csv")
    parser.add_argument("--data_dir", type=str, default="/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset")
    parser.add_argument("--output", type=str, default=f"/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/{date_str}")
    parser.add_argument("--csv", type=str, default=f"/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/{date_str}/benchmark_log.csv")
    parser.add_argument("--models", nargs="+", default=["fluxkontextdev"])
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    # ["fluxcontrolbeta", "stablediffusionv2", "kandinsky", "stablediffusion", "stablediffusionxl", "fluxkontextdev"]

    dataset = load_local_dataset(args.dataset_csv, args.data_dir)

    for model_name in args.models:
        if model_name not in MODELS_INPAINT:
            print(f"⚠️ Not known model : {model_name}")
            continue

        print(f"▶ Benchmarking model : {model_name}")
        benchmark_dataset(
            model_name=model_name,
            inpaint_fn=MODELS_INPAINT[model_name],
            load_fn=MODELS_LOAD[model_name],
            dataset=dataset,
            result_dir=os.path.join(args.output, model_name),
            csv_path=args.csv,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()
