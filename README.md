# Inpaint Benchmark Suite

## Overview

This repository is designed to evaluate and automate workflows for image inpainting and generation using state-of-the-art techniques. It is divided into two main exercises:

1. **Benchmarking Inpainting Solutions**:
   This section focuses on comparing various inpainting techniques available in the open-source community. The goal is to identify the most production-ready solution based on rigorous evaluation criteria.

2. **Automating ComfyUI Workflows**:
   This section provides a Python-based approach to programmatically execute workflows created in ComfyUI, a web-based user interface for generating images with Diffusion Models.

## Structure

- **1_inpainting_benchmark/**:
  Contains scripts, data, and results related to benchmarking inpainting solutions. Key files include:
  - `report.pdf`: A detailed report summarizing the benchmarking results.
  - `analysis.ipynb`: A notebook for analyzing benchmark results.
  - `data/`: Includes datasets and prompts used for testing.
  - `scripts/`: Contains Python scripts for running benchmarks and processing results.

- **2_comfy_runner/**:
  Focuses on automating ComfyUI workflows. Key files include:
  - `comfy2py.py`: A script to execute ComfyUI workflows programmatically.
  - `workflow_api.json`: Defines the workflow pipeline.
  - `comfyui/`: Contains the ComfyUI server and related files.

This structure ensures a clear separation between benchmarking tasks and automation workflows, making it easy to navigate and extend the repository.
