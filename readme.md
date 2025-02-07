YOLOv8  with Config-Based Training
This repository provides a YOLOv8-based model and training script for detecting bugs (or other objects) in images. It uses a separate config file (train_config.yaml) to specify hyperparameters, file paths, and other settings, making the project more user-friendly and customizable.

Project Setup and Usage
Clone the Repository

bash
Copy
Edit
git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo
Create a Python Virtual Environment (Optional)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
This installs PyTorch (with GPU support if you choose the correct wheel), Ultralytics for YOLOv8, PyYAML, and any other required packages.

Update the Configuration
In train_config.yaml, you can specify:

yaml
Copy
Edit
model_weights: yolov8n.pt            # YOLOv8 model (n, s, m, etc.)
data_yaml: BugTrackerWithHomeV2.v2i.yolov8-obb/data.yaml  # Path to your dataset config
epochs: 150                          # Number of training epochs
imgsz: 1024                          # Image size (width x height)
batch_size: 16                       # Batch size
lr0: 0.01                            # Initial learning rate
run_name: train_run4                 # Name for the run; results go in runs/detect/train_run4
save_best: true                      # Automatically copy best.pt to project root after training
Train the Model

bash
Copy
Edit
python main.py
The script checks if a GPU is available. If so, training runs on GPU; otherwise it defaults to CPU.
By default, all results (including logs, charts, etc.) are saved under runs/detect/<run_name>.
Check the Best Weights

If save_best is set to true, your best-performing weights (best.pt) will be copied to the project’s root directory after training completes.
Folder Structure
graphql
Copy
Edit
.
├── main.py                  # Main training script
├── train_config.yaml        # YAML config for training parameters
├── BugTrackerWithHomeV2.v2i.yolov8-obb/
│   ├── data.yaml            # YOLO format dataset config
│   └── ...                  # Your dataset (images, labels, etc.)
├── requirements.txt         # Dependencies
├── README.md                # You're here!
└── ...
Tips & Troubleshooting
GPU Not Detected: Verify you installed a GPU-enabled version of PyTorch and that torch.cuda.is_available() returns True.
Data Path Issues: Double-check your data.yaml paths and ensure they point to the correct folders for training and validation images.
Out of Memory Errors: Lower the batch_size or imgsz in train_config.yaml if your GPU runs out of memory.
Customization: Feel free to add additional config fields in train_config.yaml and modify main.py to support them.