# Comfy Runner

## Introduction

This project demonstrates how to programmatically run a ComfyUI workflow using Python. ComfyUI is a web-based user interface designed for generating images with Diffusion Models. It allows users to quickly iterate on an end-to-end inference pipeline.

Once the pipeline is finalized, this project provides a way to execute the workflow via a Python command line script instead of using the WebUI.

The main script, `comfy2py.py`, enables users to call any ComfyUI workflow with the following command:
```bash
python comfy2py.py --prompt="your prompt here" --neg-prompt="your negative prompt here" --steps=50
```

The script works as follows:

1. It takes a workflow designed in JSON format (`workflow_api.json`).
2. Sends the workflow to the backend via the `/prompt` endpoint.
3. Waits for the job to finish by polling the backend.
4. Once completed, it copies the resulting images to the `output` folder.


This approach is ideal for automating workflows and integrating them into larger systems.

## Installation

1. **Navigate to the `comfy` subdirectory**:
   ```bash
   cd comfy
   ```

2. **Install Dependencies**:
   Use `uv` to install dependencies:
   ```bash
   uv sync
   ```

3. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

## Running the Comfy Server

1. Navigate to the `comfyui` directory:
   ```bash
   cd comfyui
   ```

2. Start the Comfy server:
   ```bash
   python main.py
   ```

   The server will start and listen on `http://127.0.0.1:8188`.

## Preparing the Workflow

1. Create a `workflow_api.json` file in the `2_comfy_runner` directory.
   This file should define the workflow for the generation pipeline.

## Running the Automation Script

1. Navigate to the `2_comfy_runner` directory:
   ```bash
   cd 2_comfy_runner
   ```

2. Run the `comfy2py.py` script:
   ```bash
   python comfy2py.py --prompt "Your positive prompt" --neg-prompt "Your negative prompt" --steps 20
   ```

   Replace `"Your positive prompt"` and `"Your negative prompt"` with your desired prompts. Adjust the `--steps` argument as needed.

   This script automates the generation pipeline by interacting with the Comfy server and downloading the generated images.

