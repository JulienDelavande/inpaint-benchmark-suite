import os
import csv
import json
import argparse
import torch
from PIL import Image
import model_fluxcontrolbeta

# Dictionnaire de modèles
MODELS = {
    "fluxcontrolbeta": model_fluxcontrolbeta.inpaint,
}

# Paramètres modèles à logguer
MODEL_CONFIGS = {
    "fluxcontrolbeta": {"version": "beta", "source": "FluxControlNet"},
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

def benchmark_dataset(model_name, model_fn, dataset, result_dir, csv_path, num_samples=1):
    ensure_dir(result_dir)
    header = ["model_name", "image", "prompt", "image_generated", "duration_sec", "memory_MB", "model_params"]

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("RGB")
        prompt = sample["prompt"]
        image_name = os.path.basename(sample["image_path"]).split('.')[0]
        fname = f"{image_name}.png"

        mask = mask.convert("L")  # convert to grayscale
        threshold = 253
        mask = mask.point(lambda p: 255 if p > threshold else 0).convert("RGB")


        # CUDA timing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        starter.record()

        result, mask = model_fn(image, mask, prompt)

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
        mask.save(os.path.join(result_dir, f"mask_{fname}"))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, default="/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset/inpainting_prompts.csv")
    parser.add_argument("--data_dir", type=str, default="/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/dataset")
    parser.add_argument("--output", type=str, default="/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results")
    parser.add_argument("--csv", type=str, default="/fsx/jdelavande/inpaint-benchmark-suite/1_inpainting_benchmark/data/results/benchmark_log.csv")
    parser.add_argument("--models", nargs="+", default=["fluxcontrolbeta"])
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    dataset = load_local_dataset(args.dataset_csv, args.data_dir)

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"⚠️ Modèle inconnu : {model_name}")
            continue

        print(f"▶ Benchmark du modèle : {model_name}")
        benchmark_dataset(
            model_name=model_name,
            model_fn=MODELS[model_name],
            dataset=dataset,
            result_dir=os.path.join(args.output, model_name),
            csv_path=args.csv,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()
