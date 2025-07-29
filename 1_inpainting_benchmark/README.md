# Inpainting Benchmark Suite

## Overview
This project is dedicated to benchmarking various models for inpainting tasks. Inpainting refers to the process of reconstructing missing or corrupted parts of an image in a visually plausible way. The suite provides tools and scripts to evaluate the performance of different inpainting models under various conditions.

## Features
- **Dataset**: Includes a collection of prompts and results for inpainting tasks.
- **Scripts**: Contains benchmarking scripts for running and analyzing model performance.
- **Models**: Benchmarks popular inpainting models such as Stable Diffusion, ControlNet, and others.
- **Analysis**: Provides tools for analyzing and visualizing benchmarking results.

## Structure
- `report.pdf`: A detailed report summarizing the benchmarking results.
- `data/`: Contains datasets and results.
  - `dataset/inpainting_prompts.csv`: Prompts used for inpainting tasks.
  - `results/`: Stores benchmarking results.
- `scripts/`: Includes scripts for running benchmarks and model pipelines.
- `analysis.ipynb`: Jupyter notebook for analyzing benchmarking results.

## Installation

To install the dependencies for this project, use the following command:

```bash
uv sync
source .venv/bin/activate
```

This will ensure all required packages are installed as specified in the `pyproject.toml` file.

## Running the Benchmark

To run the benchmark, execute the following command:

```bash
python scripts/run_benchmark.py --dataset_csv <path_to_dataset_csv> --data_dir <path_to_data_directory> --output <path_to_output_directory> --csv <path_to_csv_log> --models <model_names> --num_samples <number_of_samples>
```

### Example

```bash
python scripts/run_benchmark.py --dataset_csv data/dataset/inpainting_prompts.csv --data_dir data/dataset --output data/results/benchmark_output --csv data/results/benchmark_output/benchmark_log.csv --models stablediffusionv2 kandinsky --num_samples 5
```

Replace `<path_to_dataset_csv>`, `<path_to_data_directory>`, `<path_to_output_directory>`, `<path_to_csv_log>`, `<model_names>`, and `<number_of_samples>` with the appropriate values.

## How to Use
1. **Setup**: Ensure all dependencies are installed as specified in `pyproject.toml`.
2. **Run Benchmarks**: Use the scripts in the `scripts/` directory to execute benchmarks for specific models.
3. **Analyze Results**: Open `analysis.ipynb` to visualize and interpret the benchmarking results.

## Models Benchmarked
- Stable Diffusion
- Stable Diffusion XL
- FluxControlNet
- Kandinsky
- FluxKontextDev