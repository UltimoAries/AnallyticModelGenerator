import os
import shutil
import torch
import yaml
from ultralytics import YOLO


def main():
    # --- GPU/CPU CHECK ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Load config from YAML ---
    script_dir = os.path.dirname(__file__)  # Directory of this script
    config_path = os.path.join(script_dir, "train_config.yaml")  # Path to the config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find the config file at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  # Parses YAML into a Python dict

    # --- Step 2: Load the YOLOv8 model ---
    model_weights = config["model_weights"]
    print(f"Loading model weights from: {model_weights}")
    model = YOLO(model_weights)

    # --- Step 3: Validate data.yaml path ---
    data_yaml_path = os.path.join(script_dir, config["data_yaml"])
    print(f"Looking for data.yaml at: {data_yaml_path}")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    # --- Step 4: Train with parameters from config ---
    print("Starting training with config parameters:")
    print(f"  epochs   = {config['epochs']}")
    print(f"  imgsz    = {config['imgsz']}")
    print(f"  batch    = {config['batch_size']}")
    print(f"  lr0      = {config['lr0']}")
    print(f"  run_name = {config['run_name']}")
    print()

    model.train(
        data=data_yaml_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch_size"],
        lr0=config["lr0"],
        name=config["run_name"],
        device=device  # <-- Applies the GPU/CPU setting we detected above
    )

    # --- Step 5: Optionally copy best.pt after training ---
    if config.get("save_best", False):
        run_dir = os.path.join("runs", "detect", config["run_name"], "weights")
        best_pt_src = os.path.join(run_dir, "best.pt")
        best_pt_dest = os.path.join(script_dir, "best.pt")

        if os.path.exists(best_pt_src):
            shutil.copy2(best_pt_src, best_pt_dest)
            print(f"Copied best.pt to {best_pt_dest}")
        else:
            print("best.pt not found! Check if training completed successfully.")


if __name__ == "__main__":
    main()
