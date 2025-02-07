# AnalyticModelGenerator - GUI Image Annotation and YOLOv8 Training Tool

---

[![Project Status](https://img.shields.io/badge/Status-In%20Development-yellow)](https://github.com/UltimoAries/AnalyticModelGenerator)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Replace with your chosen license badge if different -->

<!-- Add a project logo or screenshot here if you have one (optional) -->
<!-- ![Project Logo](path/to/your/logo.png) -->
Screenshot of the Annotation Tool UI(![image](https://github.com/user-attachments/assets/d7cb4c80-9373-438a-a293-1fdcda3f679a) <!-- Replace with a link to a screenshot of your UI -->

---


## Project Overview

**AnalyticModelGenerator** is a user-friendly, GUI-based image annotation tool designed to streamline the process of creating high-quality datasets for object detection models, with a particular focus on YOLOv8.  It aims to simplify the entire workflow from image annotation to model training, all within a single application.

**Key Features:**

*   **Intuitive GUI Annotation Interface:**
    *   Easily load images from folders.
    *   Draw bounding boxes with a user-friendly click-and-drag interface.
    *   Select object classes from a dropdown menu.
    *   Visualize annotations with color-coded bounding boxes and class labels.
    *   Zoom and pan functionalities for precise annotation.
    *   Keyboard shortcuts for faster annotation and navigation.
    *   Class editor to manage object classes (add, edit, delete classes and their colors).
*   **Dataset Formatting and Export:**
    *   Exports annotations in the widely used YOLOv8 format (`.txt` label files).
    *   Organizes datasets into `train`, `valid`, and `test` folders with `images` and `labels` subdirectories.
    *   Allows users to set percentages for train/validation/test splits.
    *   Generates `data.yaml` configuration file required for YOLOv8 training.
    *   Option to set a default save directory for annotations and exported datasets.
*   **Integrated YOLOv8 Training:**
    *   Configure YOLOv8 training parameters directly within the UI (model weights, epochs, image size, batch size, learning rate, run name, save best model).
    *   Start YOLOv8 training in a background process from the "Training" tab.
    *   Monitor training progress in a built-in console within the application.
    *   Generates `train_config.yaml` file to store training settings.
    *   Option to choose from pre-defined YOLOv8 model weights (nano, small, medium, large, xlarge).
*   **Settings Persistence:**
    *   Remembers default save directory and class configurations between application sessions.
*   **Cross-Platform Compatibility (PyQt5):**  Designed to be cross-platform, running on Windows, macOS, and Linux (where PyQt5 is supported).

**Benefits:**

*   **Streamlined Workflow:**  Combines annotation, dataset preparation, and training setup into a single tool, saving time and effort.
*   **User-Friendly GUI:**  Provides an intuitive graphical interface, making image annotation and training accessible to users without extensive command-line or coding experience.
*   **YOLOv8 Focused:** Specifically designed for creating datasets and training models for the popular YOLOv8 object detection framework.
*   **Efficient Annotation:**  Features like zoom/pan, keyboard shortcuts, and class management enhance annotation efficiency.
*   **Dataset Standardization:** Ensures consistent dataset format and structure, ready for YOLOv8 training.
*   **Training Progress Monitoring:** Built-in console allows users to track training directly within the application.

**Target Audience:**

This tool is ideal for:

*   Researchers and developers working on object detection tasks using YOLOv8.
*   Computer vision engineers needing a streamlined annotation and training pipeline.
*   Students and hobbyists learning about object detection and YOLO models.
*   Anyone who needs to quickly create and train object detection models from image datasets.

---

## Installation

Follow these steps to set up and run the AnalyticModelGenerator tool:

1.  **Prerequisites:**
    *   **Python 3.9 or later:**  Make sure you have Python 3.9 or a newer version installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
    *   **pip:** Python's package installer (`pip`) should be included with your Python installation.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/UltimoAries/AnalyticModelGenerator.git
    cd AnalyticModelGenerator
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```

4.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **On macOS and Linux:**
        ```bash
        source .venv/bin/activate
        ```
        (You should see `(.venv)` at the beginning of your command prompt after activation.)

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install all the necessary Python libraries listed in the `requirements.txt` file, including:
    *   `PyQt5` (for the GUI)
    *   `opencv-python` (for image processing)
    *   `PyYAML` (for YAML configuration files)
    *   `ultralytics` (for YOLOv8 integration)
    *   `torch`, `torchvision`, `torchaudio` (PyTorch and related libraries - for YOLOv8 training)

6.  **Run the Application:**
    ```bash
    python main.py  # Or python Test.py if you haven't renamed it yet
    ```
    This will launch the AnalyticModelGenerator GUI.

**Optional: GPU Support for Training (CUDA)**

If you have an NVIDIA GPU and want to enable GPU acceleration for YOLOv8 training (highly recommended for faster training), you need to:

1.  **Install NVIDIA CUDA Toolkit:** Download and install the CUDA Toolkit compatible with your NVIDIA GPU driver from [NVIDIA Developer website](https://developer.nvidia.com/cuda-toolkit-archive).
2.  **Install PyTorch with CUDA Support:**  When installing dependencies (step 5 above), use the appropriate `pip install` command for PyTorch with CUDA support.  Visit the [PyTorch website](https://pytorch.org/) and select your PyTorch version, OS, package manager (pip), Python version, and CUDA version to get the correct command.  It will look something like:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Example for CUDA 11.8
    ```
    (Replace `cu118` with your CUDA version if needed).

---

## Usage Instructions

1.  **Annotation Tab:**
    *   **Load Images:** Click "Load Folder" to select a folder containing images you want to annotate.
    *   **Navigate Images:** Use "Previous" and "Next" buttons (or Left/Right arrow keys) to navigate through images.
    *   **Zoom:** Use the Zoom slider or Ctrl + Mouse Wheel to zoom in/out. Middle mouse button drag to pan.
    *   **Select Class:** Choose a class from the "Class" dropdown combo box. Edit classes using the "Edit Classes" button.
    *   **Draw Bounding Boxes:** Click and drag on the image to draw bounding boxes around objects.
    *   **Select and Delete Boxes:** Click inside a bounding box to select it (dashed line). Press Delete key to delete the selected box.
    *   **Save Annotations:** Click "Save Annotations" to save annotations for the current image, or "Save All Annotations" to save for all images in the loaded folder. Annotations are saved in YOLO format in a `labels` subfolder (in the default save directory or image folder).

2.  **Training Tab:**
    *   **Export Dataset Settings:**
        *   Set "Train %", "Validation %", and "Test %" to define dataset splits.
        *   Choose an "Export Directory" using "Browse...".
        *   Click "Export Dataset" to export the annotated dataset into the specified directory, creating `train`, `valid`, `test` folders with `images` and `labels` subfolders, and generating `data.yaml` and `train_config.yaml`.
    *   **Training Configuration:**
        *   Select "Model Weights" (YOLOv8n, yolov8s, etc.) from the dropdown.
        *   Adjust "Epochs", "Image Size", "Batch Size", "Learning Rate", "Run Name" as needed.
        *   Check "Save Best Model" to save the best model weights during training.
        *   **Start Training:** Once a dataset is exported, the "Start Training" button will be enabled. Click it to begin YOLOv8 training in the background. Monitor the training progress in the console area below the button.

3.  **Settings Tab:**
    *   **Default Save Directory:** Set the default directory where annotation labels and exported datasets will be saved using "Browse...". This setting is persistent across application sessions.

---

## Configuration Files

*   **`classes.yaml`:**  Stores the object class names and their associated colors. This file is created and updated when you use the "Edit Classes" dialog in the "Annotation" tab.
*   **`settings.txt`:**  Stores application settings, currently just the "Default Save Directory".
*   **`train_config.yaml` (Generated during dataset export):** Stores the training configuration parameters (model weights, data.yaml path, epochs, image size, batch size, learning rate, run name, save best model) as set in the "Training" tab UI. This file is used by `train_script.py` to configure the YOLOv8 training process.
*   **`data.yaml` (Generated during dataset export):**  A standard YOLOv8 data configuration file that defines the paths to your training, validation, and test datasets, the number of classes, and class names.

---

## Training Script (`train_script.py`)

The `train_script.py` file contains the core YOLOv8 training logic using the `ultralytics` library. It is launched in a separate process when you click "Start Training" in the application. It loads training parameters from `train_config.yaml` and performs the YOLOv8 training.

---

## Contributing (Optional)

<!-- If you are open to contributions, add guidelines here -->
<!--
If you'd like to contribute to this project, please feel free to fork the repository and submit pull requests.
You can also report bugs or suggest new features by opening issues.
-->

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgments (Optional)

<!-- If you want to acknowledge any libraries, resources, or people, list them here -->
<!--
* PyQt5 - For the cross-platform GUI framework.
* Ultralytics YOLOv8 - For the powerful object detection framework.
* OpenCV - For image processing functionalities.
* PyYAML - For YAML file handling.
-->

---

## Future Features (Optional - if you want to mention planned features)

<!-- List any planned future features or improvements here -->
<!--
* Implementation of the "Testing" tab for model evaluation.
* More advanced annotation tools (e.g., polygon annotation).
* Real-time object detection demo within the application.
* Support for different annotation formats (e.g., COCO, Pascal VOC).
-->

---

**This `README.md` is a starting point. Please customize it further with:**

*   **More details about *your specific tool* and its unique aspects.**
*   **Clearer and more detailed instructions if needed.**
*   **High-quality screenshots and potentially a demo GIF or video to make it visually appealing.**
*   **Your chosen license information in the `LICENSE` file and README badge.**

Let me know if you'd like me to refine any specific section or add more details!
