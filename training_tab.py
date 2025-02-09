# training_tab.py
import os
import sys
import random
import shutil
import yaml
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QLabel, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QTextEdit,
                             QFileDialog, QComboBox, QMessageBox)
from PyQt5.QtCore import QProcess

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()

        # --- UI Elements ---
        self.training_console = QTextEdit(self)
        self.training_console.setReadOnly(True)

        # Export Dataset Group
        self.export_dataset_group_layout = QFormLayout()
        self.train_percent_label = QLabel("Train %:", self)
        self.train_percent_spinbox = QSpinBox(self)
        self.train_percent_spinbox.setRange(0, 100); self.train_percent_spinbox.setValue(70)
        self.valid_percent_label = QLabel("Validation %:", self)
        self.valid_percent_spinbox = QSpinBox(self)
        self.valid_percent_spinbox.setRange(0, 100); self.valid_percent_spinbox.setValue(20)
        self.test_percent_label = QLabel("Test %:", self)
        self.test_percent_spinbox = QSpinBox(self)
        self.test_percent_spinbox.setRange(0, 100); self.test_percent_spinbox.setValue(10)
        self.export_dir_label = QLabel("Export Directory:", self)
        self.export_dir_edit = QLineEdit(self)
        self.export_dir_browse_button = QPushButton("Browse...", self)
        self.export_dir_browse_button.clicked.connect(self.browse_export_dir)
        self.export_dataset_button = QPushButton("Export Dataset", self)
        self.export_dataset_button.clicked.connect(self.export_dataset)
        self.export_dataset_button.setEnabled(False) # Start Disabled

        self.export_dataset_group_layout.addRow(self.train_percent_label, self.train_percent_spinbox)
        self.export_dataset_group_layout.addRow(self.valid_percent_label, self.valid_percent_spinbox)
        self.export_dataset_group_layout.addRow(self.test_percent_label, self.test_percent_spinbox)
        self.export_dataset_group_layout.addRow(self.export_dir_label, self.export_dir_edit)
        self.export_dataset_group_layout.addRow(self.export_dir_browse_button)
        self.export_dataset_group_layout.addRow(self.export_dataset_button)


        # Training Configuration Group
        self.training_config_group_layout = QFormLayout()
        self.training_config_group_layout.addRow(QLabel("<b>Training Configuration</b>"), QLabel(""))
        self.model_weights_label = QLabel("Model Weights:", self)
        self.model_weights_combo = QComboBox(self)
        self.model_weights_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_weights_combo.setCurrentText("yolov8n.pt")
        self.epochs_label = QLabel("Epochs:", self)
        self.epochs_spinbox = QSpinBox(self)
        self.epochs_spinbox.setRange(1, 1000); self.epochs_spinbox.setValue(150)
        self.imgsz_label = QLabel("Image Size:", self)
        self.imgsz_spinbox = QSpinBox(self)
        self.imgsz_spinbox.setRange(256, 2048); self.imgsz_spinbox.setValue(640)
        self.batch_size_label = QLabel("Batch Size:", self)
        self.batch_size_spinbox = QSpinBox(self)
        self.batch_size_spinbox.setRange(1, 128); self.batch_size_spinbox.setValue(16)
        self.lr0_label = QLabel("Initial Learning Rate:", self)
        self.lr0_doublespinbox = QDoubleSpinBox(self)
        self.lr0_doublespinbox.setRange(0.00001, 0.1); self.lr0_doublespinbox.setSingleStep(0.001); self.lr0_doublespinbox.setValue(0.01)
        self.run_name_label = QLabel("Run Name:", self)
        self.run_name_edit = QLineEdit(self)
        self.run_name_edit.setText("train_run1")
        self.save_best_checkbox = QCheckBox("Save Best Model", self)
        self.save_best_checkbox.setChecked(True)

        self.training_config_group_layout.addRow(self.model_weights_label, self.model_weights_combo)
        self.training_config_group_layout.addRow(self.epochs_label, self.epochs_spinbox)
        self.training_config_group_layout.addRow(self.imgsz_label, self.imgsz_spinbox)
        self.training_config_group_layout.addRow(self.batch_size_label, self.batch_size_spinbox)
        self.training_config_group_layout.addRow(self.lr0_label, self.lr0_doublespinbox)
        self.training_config_group_layout.addRow(self.run_name_label, self.run_name_edit)
        self.training_config_group_layout.addRow(self.save_best_checkbox, QLabel("")) # Empty label for alignment


        self.start_training_button = QPushButton("Start Training", self)
        self.start_training_button.clicked.connect(self.start_training)
        self.start_training_button.setEnabled(False) # Initially disabled

        # --- Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(self.export_dataset_group_layout)
        main_layout.addLayout(self.training_config_group_layout) # Add training config
        main_layout.addWidget(self.start_training_button)
        main_layout.addWidget(self.training_console)
        self.setLayout(main_layout) # Set the main layout. Very important!

        # --- Data ---
        self.process = None   # To store the QProcess instance
        self.image_paths = [] # Add the missing self.imagepaths
        self.classes = {}     # Add in the missing self.classes

    def browse_export_dir(self):
        """Opens a dialog to select the export directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if dir_path:
            self.export_dir_edit.setText(dir_path)

    def export_dataset(self):
      """Exports the labeled data in YOLO format."""
      export_dir = self.export_dir_edit.text()
      train_percent = self.train_percent_spinbox.value()
      valid_percent = self.valid_percent_spinbox.value()
      test_percent = self.test_percent_spinbox.value()

      if not export_dir:
          QMessageBox.warning(self, "Warning", "Please select an export directory.")
          return

      if not self.image_paths:
          QMessageBox.warning(self, "Warning", "No images loaded to export.")
          return

      if train_percent + valid_percent + test_percent != 100:
          QMessageBox.warning(self, "Warning", "Train, Validation, and Test percentages must sum to 100%.")
          return

      # Create export directories
      try:
          os.makedirs(export_dir, exist_ok=True)
          train_img_dir = os.path.join(export_dir, "train", "images")
          train_label_dir = os.path.join(export_dir, "train", "labels")
          valid_img_dir = os.path.join(export_dir, "valid", "images")
          valid_label_dir = os.path.join(export_dir, "valid", "labels")
          test_img_dir = os.path.join(export_dir, "test", "images")
          test_label_dir = os.path.join(export_dir, "test", "labels")
          os.makedirs(train_img_dir, exist_ok=True)
          os.makedirs(train_label_dir, exist_ok=True)
          os.makedirs(valid_img_dir, exist_ok=True)
          os.makedirs(valid_label_dir, exist_ok=True)
          os.makedirs(test_img_dir, exist_ok=True)
          os.makedirs(test_label_dir, exist_ok=True)
      except OSError as e:
          QMessageBox.critical(self, "Error", f"Error creating export directories: {e}")
          return

      # Split image paths into train, valid, and test sets
      random.shuffle(self.image_paths)  # Shuffle for random split
      num_images = len(self.image_paths)
      train_split = int(num_images * train_percent / 100)
      valid_split = int(num_images * valid_percent / 100)

      train_images = self.image_paths[:train_split]
      valid_images = self.image_paths[train_split : train_split + valid_split]
      test_images = self.image_paths[train_split + valid_split:]

      print(f"Train images: {len(train_images)}, Valid images: {len(valid_images)}, Test images: {len(test_images)}")

      # Copy images and labels to the respective directories
      image_sets = [
          (train_images, train_img_dir, train_label_dir),
          (valid_images, valid_img_dir, valid_label_dir),
          (test_images, test_img_dir, test_label_dir),
      ]

      for image_list, img_dir, label_dir in image_sets:
          for image_path in image_list:
              try:
                  # Copy image
                  filename = os.path.basename(image_path)
                  dest_image_path = os.path.join(img_dir, filename)
                  shutil.copy2(image_path, dest_image_path) # Copy with metadata

                  # Copy/Create label file (if annotations exist)
                  if image_path in self.image_boxes:
                      boxes = self.image_boxes[image_path]
                      if boxes: # Only if there are any boxes annotated.
                          label_filename = os.path.splitext(filename)[0] + ".txt"
                          dest_label_path = os.path.join(label_dir, label_filename)
                          self.save_annotation_to_path(image_path, dest_label_path, boxes)  #Need to define
              except Exception as e:
                  QMessageBox.critical(self, "Error", f"Error copying files: {e}")
                  return  # Stop on error

      # Create data.yaml file
      data_yaml_path = os.path.join(export_dir, "data.yaml")
      data_yaml_content = {
          'train': os.path.relpath(train_img_dir, export_dir).replace("\\", "/"),  # Use relative paths
          'val': os.path.relpath(valid_img_dir, export_dir).replace("\\", "/"),    # and forward slashes
          'test': os.path.relpath(test_img_dir, export_dir).replace("\\", "/"),   # for cross-platform compatibility
          'nc': len(self.classes),
          'names': list(self.classes.keys())
      }
      try:
          with open(data_yaml_path, 'w') as outfile:
              yaml.dump(data_yaml_content, outfile, default_flow_style=False)
      except Exception as e:
           QMessageBox.critical(self, "Error", f"Error writing data.yaml: {e}")
           return

      # Create train_config.yaml
      train_config_path = os.path.join(export_dir, "train_config.yaml")
      train_config_content = {
          'model_weights': self.model_weights_combo.currentText(),  # Get selected model
          'data_yaml': os.path.join(".", "data.yaml").replace("\\", "/"),  # Relative path
          'epochs': self.epochs_spinbox.value(),      # Get values from spinboxes
          'imgsz': self.imgsz_spinbox.value(),
          'batch_size': self.batch_size_spinbox.value(),
          'lr0': self.lr0_doublespinbox.value(),
          'run_name': self.run_name_edit.text(),
          'save_best': self.save_best_checkbox.isChecked()
      }
      try:
          with open(train_config_path, 'w') as outfile:
              yaml.dump(train_config_content, outfile, default_flow_style=False)
      except Exception as e:
          QMessageBox.critical(self, "Error", f"Error writing train_config.yaml: {e}")
          return

      QMessageBox.information(self, "Success", "Dataset exported successfully with training configuration!")

      # Enable the Start Training button
      self.start_training_button.setEnabled(True)

    def save_annotation_to_path(self, image_path, label_file_path, boxes):
      """Saves annotations for a single image to a specified path."""
      img = cv2.imread(image_path)
      if img is None:
          print(f"Warning: Could not read image to get dimensions: {image_path}")
          return

      height, width, _ = img.shape

      with open(label_file_path, "w+") as f:  # Open for writing (and create if it doesn't exist)
          for x1, y1, x2, y2, class_name in boxes:
              class_id = list(self.classes.keys()).index(class_name)

              # Normalize coordinates (YOLO format)
              x_center = ((x1 + x2) / 2) / width
              y_center = ((y1 + y2) / 2) / height
              box_width = (x2 - x1) / width
              box_height = (y2 - y1) / height

              f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


    def start_training(self):
        """Starts the YOLOv8 training process in a separate QProcess."""
        export_dir = self.export_dir_edit.text()
        if not export_dir:
            QMessageBox.warning(self, "Warning", "Please select an export directory first.")
            return

        train_config_path = os.path.join(export_dir, "train_config.yaml")
        if not os.path.exists(train_config_path):
            QMessageBox.critical(self, "Error", f"train_config.yaml not found in {export_dir}.  Please export the dataset.")
            return


        # Disable the Start Training button while training is in progress.
        self.start_training_button.setEnabled(False)
        QApplication.processEvents() # Process any pending events to keep UI responsive.

        # Create a QProcess
        self.process = QProcess(self)

        # OPTIONAL: Merge stdout and stderr into a single stream.  Easier to handle.
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        # Connect signals BEFORE starting the process
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)  # Not used if merging channels
        self.process.finished.connect(self.training_finished)
        self.process.errorOccurred.connect(self.on_process_error)


        # Build the command to run.  Use a separate script (train_script.py)
        train_script_fullpath = os.path.join(os.path.dirname(__file__), "train_script.py")
        command = [
            sys.executable,  # Use the same Python interpreter as the GUI
            "-u", # Unbuffered output.  VERY IMPORTANT for live output.
            train_script_fullpath,
            "--config",
            train_config_path,
        ]

        self.training_console.append("Starting training process...")  # Debug message
        self.process.start(command[0], command[1:])  # Start the process
        self.training_console.append("Process started.  Waiting for output...")

    def handle_stdout(self):
      """Handles standard output from the training process."""
      # If MergedChannels is used, both stdout and stderr will come through here
      output = self.process.readAllStandardOutput().data().decode()
      self.training_console.append(output)

    def handle_stderr(self):
      """Handles standard error from the training process."""
      # Only used if NOT merging channels
      error_output = self.process.readAllStandardError().data().decode()
      self.training_console.append(f"<span style='color:red;'>{error_output}</span>")  # Display in red

    def training_finished(self, exitCode, exitStatus):
      """Called when the training process finishes."""
      self.start_training_button.setEnabled(True)  # Re-enable the button
      self.training_console.append(f"Training process finished with exit code {exitCode}")
      QMessageBox.information(self, "Training Finished", "Training process has finished.")
      self.process = None # Cleanup

    def on_process_error(self, error):
      """Handles errors that occur when starting or running the process."""
      self.training_console.append(f"<span style='color:red;'>Process error occurred: {error}</span>")
      self.start_training_button.setEnabled(True) # Re-enable button
      self.process = None # Cleanup

    def set_image_paths(self, image_paths):  #Added to pass image paths
        """Sets the image paths for the training tab (used during export)."""
        self.image_paths = image_paths

    def set_classes(self, classes): # Added to pass classes data
      """Sets the classes dictionary (used during export)."""
      self.classes = classes
    def set_image_boxes(self, image_boxes):  # Added
        """Sets the image_boxes data (used during export)."""
        self.image_boxes = image_boxes