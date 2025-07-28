import argparse
import json
import requests
import uuid
import os
import time

COMFY_API_URL = "http://127.0.0.1:8188"
WORKFLOW_FILE = "workflow_api.json"  # le fichier JSON exporté depuis ComfyUI en format API

def load_workflow():
    with open(WORKFLOW_FILE, "r") as f:
        return json.load(f)

def update_prompt(workflow, prompt, negative_prompt, steps):
    prompt_data = workflow["prompt"]
    for node_id, node in prompt_data.items():
        if node["class_type"] == "CLIPTextEncode":
            if "positive" in node_id or node["inputs"].get("text", "").startswith("masterpiece"):
                node["inputs"]["text"] = prompt
            elif "negative" in node_id or node["inputs"].get("text", "").startswith("bad"):
                node["inputs"]["text"] = negative_prompt
        elif node["class_type"] == "KSampler":
            node["inputs"]["steps"] = steps
    return workflow

def queue_workflow(workflow):
    payload = {
        "prompt": workflow["prompt"],
        "client_id": str(uuid.uuid4())
    }
    response = requests.post(f"{COMFY_API_URL}/prompt", json=payload)
    response.raise_for_status()
    return response.json()["prompt_id"]  # ← important : utilise le vrai prompt_id


def wait_for_result(prompt_id):
    while True:
        res = requests.get(f"{COMFY_API_URL}/history/{prompt_id}")
        if res.status_code == 200:
            data = res.json()
            if prompt_id in data:
                prompt_data = data[prompt_id]
                print(prompt_data.get("status", {}))
                if prompt_data.get("status", {}).get("completed", False):
                    return prompt_data
        time.sleep(1)


def download_images(output_data):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for node_id, node_data in output_data.get("outputs", {}).items():
        for image in node_data.get("images", []):
            filename = image["filename"]
            subfolder = image.get("subfolder", "")
            image_url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
            img_data = requests.get(image_url).content
            save_path = os.path.join(output_dir, filename)
            with open(save_path, "wb") as f:
                f.write(img_data)
            print(f"Saved image to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Positive prompt")
    parser.add_argument("--neg-prompt", required=True, help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    args = parser.parse_args()

    workflow = load_workflow()
    updated_workflow = update_prompt(workflow, args.prompt, args.neg_prompt, args.steps)
    client_id = queue_workflow(updated_workflow)
    print(f"Waiting for generation with client_id: {client_id}")
    output_data = wait_for_result(client_id)
    download_images(output_data)

if __name__ == "__main__":
    main()
