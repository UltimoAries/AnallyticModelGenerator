#train_script
import os
import shutil
import torch
import yaml
from ultralytics import YOLO
import argparse

def get_incremented_run_name(base_name):
    """
    Generates an incremented run name (e.g., runs, runs2, runs3, etc.) if the base name already exists.
    """
    run_name = base_name
    counter = 1
    while os.path.exists(os.path.join("runs", "detect", run_name)):
        run_name = f"{base_name}{counter}"
        counter += 1
    return run_name

def main(config_path):
    # --- GPU/CPU CHECK ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Load config from YAML ---
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find the config file at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Step 2: Adjust run_name if it already exists ---
    base_run_name = config["run_name"]
    config["run_name"] = get_incremented_run_name(base_run_name)
    print(f"Adjusted run_name to: {config['run_name']}")

    # --- Step 3: Load the YOLOv8 model ---
    model_weights = config["model_weights"]
    print(f"Loading model weights from: {model_weights}")
    model = YOLO(model_weights)

    # --- Step 4: Validate data.yaml path ---
    data_yaml_path = os.path.join(os.path.dirname(config_path), config["data_yaml"])
    print(f"Looking for data.yaml at: {data_yaml_path}")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    # --- Step 5: Train with parameters from config ---
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
        device=device
    )

    # --- Step 6: Optionally copy best.pt after training ---
    if config.get("save_best", False):
        run_dir = os.path.join("runs", "detect", config["run_name"], "weights")
        best_pt_src = os.path.join(run_dir, "best.pt")
        best_pt_dest = os.path.join(os.path.dirname(config_path), "best.pt")

        if os.path.exists(best_pt_src):
            shutil.copy2(best_pt_src, best_pt_dest)
            print(f"Copied best.pt to {best_pt_dest}")
        else:
            print("best.pt not found! Check if training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to train_config.yaml")
    args = parser.parse_args()

    main(args.config)