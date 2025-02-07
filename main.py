import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QLabel, QFileDialog, QScrollArea, QHBoxLayout,
                             QSlider, QLineEdit, QDialog, QListWidget,
                             QColorDialog, QListWidgetItem, QMessageBox, QComboBox,
                             QTabWidget, QShortcut, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit) # Added QTextEdit to imports
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QColor, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtCore import QProcessEnvironment
import yaml
import random
import shutil
from PyQt5.QtCore import QProcess
import torch
import yaml
from ultralytics import YOLO

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Annotation Tool")
        self.setGeometry(100, 100, 800, 600)

        # --- Create Tab Widget ---
        self.tab_widget = QTabWidget()
        self.annotation_tab = QWidget()
        self.settings_tab = QWidget()
        self.training_tab = QWidget()
        self.testing_tab = QWidget()

        self.tab_widget.addTab(self.annotation_tab, "Annotation")
        self.tab_widget.addTab(self.training_tab, "Training")
        self.tab_widget.addTab(self.testing_tab, "Testing")
        self.tab_widget.addTab(self.settings_tab, "Settings")

        # --- Placeholder Labels for New Tabs ---
        training_placeholder_label = QLabel("<h2>Training Data Setup will go here</h2>",
                                            self.training_tab)
        training_placeholder_label.setAlignment(Qt.AlignCenter)
        self.training_tab_main_layout = QVBoxLayout(self.training_tab)  # Main layout for Training Tab
        self.training_tab_main_layout.addWidget(training_placeholder_label)

        testing_placeholder_label = QLabel("<h2>Testing/Validation Data Setup will go here</h2>",
                                           self.testing_tab)
        testing_placeholder_label.setAlignment(Qt.AlignCenter)
        testing_layout = QVBoxLayout(self.testing_tab)
        testing_layout.addWidget(testing_placeholder_label)

        # --- Training Tab UI Elements (Expanded and Layout Corrected) ---
        # --- Export Dataset UI ---
        self.export_dataset_group_layout = QFormLayout()  # Group layout for Export Dataset section

        self.train_percent_label = QLabel("Train %:", self.training_tab)
        self.train_percent_spinbox = QSpinBox(self.training_tab)
        self.train_percent_spinbox.setRange(0, 100)
        self.train_percent_spinbox.setValue(70)
        self.export_dataset_group_layout.addRow(self.train_percent_label, self.train_percent_spinbox)

        self.valid_percent_label = QLabel("Validation %:", self.training_tab)
        self.valid_percent_spinbox = QSpinBox(self.training_tab)
        self.valid_percent_spinbox.setRange(0, 100)
        self.valid_percent_spinbox.setValue(20)
        self.export_dataset_group_layout.addRow(self.valid_percent_label, self.valid_percent_spinbox)

        self.test_percent_label = QLabel("Test %:", self.training_tab)
        self.test_percent_spinbox = QSpinBox(self.training_tab)
        self.test_percent_spinbox.setRange(0, 100)
        self.test_percent_spinbox.setValue(10)
        self.export_dataset_group_layout.addRow(self.test_percent_label, self.test_percent_spinbox)

        self.export_dir_label = QLabel("Export Directory:", self.training_tab)
        self.export_dir_edit = QLineEdit(self.training_tab)
        self.export_dir_browse_button = QPushButton("Browse...", self.training_tab)
        self.export_dir_browse_button.clicked.connect(self.browse_export_dir)
        self.export_dataset_group_layout.addRow(self.export_dir_label, self.export_dir_edit)
        self.export_dataset_group_layout.addRow(self.export_dir_browse_button)

        self.export_dataset_button = QPushButton("Export Dataset", self.training_tab)
        self.export_dataset_button.clicked.connect(self.export_dataset)
        self.export_dataset_button.setEnabled(False)
        self.export_dataset_group_layout.addRow(self.export_dataset_button)

        self.training_tab_main_layout.addLayout(
            self.export_dataset_group_layout)  # Add Export Dataset group to main layout
        # --- Training Console UI ---

        self.training_console = QTextEdit(self.training_tab)  # Create QTextEdit for console
        self.training_console.setReadOnly(True)  # Make it read-only
        self.training_tab_main_layout.addWidget(self.training_console)  # Add console to main layout

        # --- Training Configuration UI ---
        self.training_config_group_layout = QFormLayout()  # Group layout for Training Config section
        self.training_config_group_layout.addRow(QLabel("<b>Training Configuration</b>"), QLabel(""))  # Section Header

        self.model_weights_label = QLabel("Model Weights:", self.training_tab)
        self.model_weights_combo = QComboBox(self.training_tab)
        self.model_weights_combo.addItems(
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_weights_combo.setCurrentText("yolov8n.pt")
        self.training_config_group_layout.addRow(self.model_weights_label, self.model_weights_combo)

        self.epochs_label = QLabel("Epochs:", self.training_tab)
        self.epochs_spinbox = QSpinBox(self.training_tab)
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(150)
        self.training_config_group_layout.addRow(self.epochs_label, self.epochs_spinbox)

        self.imgsz_label = QLabel("Image Size:", self.training_tab)
        self.imgsz_spinbox = QSpinBox(self.training_tab)
        self.imgsz_spinbox.setRange(256, 2048)
        self.imgsz_spinbox.setValue(1024)
        self.training_config_group_layout.addRow(self.imgsz_label, self.imgsz_spinbox)

        self.batch_size_label = QLabel("Batch Size:", self.training_tab)
        self.batch_size_spinbox = QSpinBox(self.training_tab)
        self.batch_size_spinbox.setRange(1, 128)
        self.batch_size_spinbox.setValue(16)
        self.training_config_group_layout.addRow(self.batch_size_label, self.batch_size_spinbox)

        self.lr0_label = QLabel("Learning Rate:", self.training_tab)
        self.lr0_doublespinbox = QDoubleSpinBox(self.training_tab)
        self.lr0_doublespinbox.setRange(0.00001, 0.1)
        self.lr0_doublespinbox.setSingleStep(0.001)
        self.lr0_doublespinbox.setValue(0.01)
        self.training_config_group_layout.addRow(self.lr0_label, self.lr0_doublespinbox)

        self.run_name_label = QLabel("Run Name:", self.training_tab)
        self.run_name_edit = QLineEdit(self.training_tab)
        self.run_name_edit.setText("train_run1")
        self.training_config_group_layout.addRow(self.run_name_label, self.run_name_edit)

        self.save_best_checkbox = QCheckBox("Save Best Model", self.training_tab)
        self.save_best_checkbox.setChecked(True)
        self.training_config_group_layout.addRow(self.save_best_checkbox, QLabel(""))  # Checkbox takes full row

        self.training_tab_main_layout.addLayout(
            self.training_config_group_layout)  # Add Training Config group to main layout

        self.start_training_button = QPushButton("Start Training", self.training_tab)
        self.start_training_button.clicked.connect(self.start_training)
        self.start_training_button.setEnabled(False)
        self.training_tab_main_layout.addWidget(self.start_training_button)

        # --- Annotation Tab UI Elements (Rest remains the same) ---
        self.load_folder_button = QPushButton("Load Folder", self.annotation_tab)
        self.load_folder_button.clicked.connect(self.load_folder)

        self.prev_button = QPushButton("Previous", self.annotation_tab)
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next", self.annotation_tab)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)

        self.image_label = QLabel(self.annotation_tab)
        self.image_label.setScaledContents(False)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.setCursor(QCursor(Qt.CrossCursor))
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea(self.annotation_tab)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        self.image_info_label = QLabel("No folder loaded", self.annotation_tab)

        self.zoom_slider = QSlider(Qt.Horizontal, self.annotation_tab)
        self.zoom_slider.setRange(1, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)

        self.zoom_label = QLabel("Zoom: 100%", self.annotation_tab)

        self.edit_classes_button = QPushButton("Edit Classes", self.annotation_tab)
        self.edit_classes_button.clicked.connect(self.open_class_editor)

        self.class_combo = QComboBox(self.annotation_tab)
        self.class_combo.currentIndexChanged.connect(self.class_selected)
        self.current_class = None

        # --- Color Preview ---
        self.color_preview = QLabel(self.annotation_tab)
        self.color_preview.setFixedSize(20, 20)
        self.update_color_preview()

        self.save_button = QPushButton("Save Annotations", self.annotation_tab)
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)

        self.save_all_button = QPushButton("Save All Annotations", self.annotation_tab)
        self.save_all_button.clicked.connect(self.save_all_annotations)
        self.save_all_button.setEnabled(False)

        # --- Annotation Tab Layout ---
        hbox = QHBoxLayout()
        hbox.addWidget(self.load_folder_button)
        hbox.addWidget(self.image_info_label)

        button_hbox = QHBoxLayout()
        button_hbox.addWidget(self.prev_button)
        button_hbox.addWidget(self.next_button)

        zoom_hbox = QHBoxLayout()
        zoom_hbox.addWidget(QLabel("Zoom:"))
        zoom_hbox.addWidget(self.zoom_slider)
        zoom_hbox.addWidget(self.zoom_label)

        class_hbox = QHBoxLayout()
        class_hbox.addWidget(QLabel("Class:"))
        class_hbox.addWidget(self.class_combo)
        class_hbox.addWidget(self.color_preview)

        vbox = QVBoxLayout(self.annotation_tab)
        vbox.addLayout(hbox)
        vbox.addLayout(button_hbox)
        vbox.addWidget(self.scroll_area)
        vbox.addLayout(zoom_hbox)
        vbox.addWidget(self.edit_classes_button)
        vbox.addLayout(class_hbox)
        vbox.addWidget(self.save_button)
        vbox.addWidget(self.save_all_button)

        # --- Settings Tab UI Elements ---
        self.default_save_dir_label = QLabel("Default Save Directory:", self.settings_tab)
        self.default_save_dir_edit = QLineEdit(self.settings_tab)
        self.browse_button = QPushButton("Browse...", self.settings_tab)
        self.browse_button.clicked.connect(self.browse_for_save_dir)

        # --- Settings Tab Layout ---
        settings_layout = QFormLayout(self.settings_tab)
        settings_layout.addRow(self.default_save_dir_label, self.default_save_dir_edit)
        settings_layout.addRow(self.browse_button)

        # --- Main Window Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        # --- Data ---
        self.image_paths = []
        self.current_image_index = -1
        self.zoom_level = 100
        self.classes = {}
        self.image_boxes = {}
        self.drawing = False
        self.start_point = QPoint()
        self.selected_box = None
        self.default_save_dir = ""
        self.load_settings()
        self.panning = False
        self.pan_start_point = QPoint()

        # --- Keyboard Shortcuts ---
        self.prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_shortcut.activated.connect(self.prev_image)
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_shortcut.activated.connect(self.next_image)
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.delete_shortcut.activated.connect(self.delete_selected_box)

        self.load_classes()

    def load_folder(self):
        """Loads images from a selected folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            if self.image_paths:
                self.current_image_index = 0
                self.load_image()
                self.update_image_info()
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                self.save_button.setEnabled(True)
                self.save_all_button.setEnabled(True)
                self.export_dataset_button.setEnabled(True)
            else:
                self.image_label.clear()
                self.image_info_label.setText("No images found in folder.")
                self.current_image_index = -1
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.save_all_button.setEnabled(False)
                self.export_dataset_button.setEnabled(False)
        else:
            QMessageBox.critical(self, "Error", f"Error loading image: Bad Path")
            self.clear_image()

    def load_image(self):
        """Loads and displays the current image."""
        if 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            try:
                cv_img = cv2.imread(image_path)
                if cv_img is None:
                    raise ValueError(f"Could not load image at {image_path}")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                height, width, channel = cv_img.shape
                bytes_per_line = 3 * width
                q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.original_pixmap = QPixmap.fromImage(q_image)

                # --- Explicitly set image_label size to pixmap size BEFORE zoom ---
                self.image_label.setFixedSize(self.original_pixmap.width(),
                                              self.original_pixmap.height())

                self.update_zoom()
                self.selected_box = None
                self.image_label.setCursor(QCursor(Qt.CrossCursor))
                h_scroll_val = self.scroll_area.horizontalScrollBar().value()
                v_scroll_val = self.scroll_area.verticalScrollBar().value()

                print(f"--- Loading Image: {os.path.basename(image_path)} ---")
                print(f"Image Dimensions: {width} x {height}")
                print(f"Pixmap Size: {self.original_pixmap.size().width()} x {self.original_pixmap.size().height()}")
                print(
                    f"Image Label Size: {self.image_label.size().width()} x {self.image_label.size().height()}")
                print(
                    f"Scroll Area Viewport Size: {self.scroll_area.viewport().size().width()} x {self.scroll_area.viewport().size().height()}")
                print(f"Scroll H: {h_scroll_val}, V: {v_scroll_val}")
                self.scroll_area.viewport().update()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {e}")
                self.clear_image()

    def clear_image(self):
        """Clears current image and pixmap"""
        self.image_label.clear()
        self.original_pixmap = None

    def next_image(self):
        """Navigates to the next image."""
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image()
            self.update_image_info()

    def prev_image(self):
        """Navigates to the previous image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
            self.update_image_info()

    def update_image_info(self):
        """Updates the image information label."""
        if self.image_paths:
            filename = os.path.basename(self.image_paths[self.current_image_index])
            self.image_info_label.setText(
                f"Folder Loaded: {os.path.dirname(self.image_paths[self.current_image_index])} | Image: {filename} | {self.current_image_index + 1}/{len(self.image_paths)}")
        else:
            self.image_info_label.setText("No folder loaded")

    def update_button_states(self):
        """Enables/disables navigation buttons based on current image index."""
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_paths) - 1)

    def update_zoom(self):
        """Updates the zoom level and redraws the image."""
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            self.zoom_level = self.zoom_slider.value()
            new_width = int(self.original_pixmap.width() * self.zoom_level / 100)
            new_height = int(self.original_pixmap.height() * self.zoom_level / 100)
            scaled_pixmap = self.original_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()
            self.update_scroll_bars()
            self.zoom_label.setText(f"Zoom: {self.zoom_level}%")

    def update_scroll_bars(self):
        """Centers the scroll bars if the image is smaller than the view."""
        scroll_area_width = self.scroll_area.viewport().width()
        scroll_area_height = self.scroll_area.viewport().height()
        image_width = self.image_label.width()
        image_height = self.image_label.height()

        horizontal_scroll_bar = self.scroll_area.horizontalScrollBar()
        vertical_scroll_bar = self.scroll_area.verticalScrollBar()

        if image_width <= scroll_area_width:
            horizontal_scroll_bar.setValue(int((scroll_area_width - image_width) / 2))
        else:
            horizontal_scroll_bar.setValue(int((horizontal_scroll_bar.maximum() - horizontal_scroll_bar.minimum()) / 2))

        if image_height <= scroll_area_height:
            vertical_scroll_bar.setValue(int((scroll_area_height - image_height) / 2))
        else:
            vertical_scroll_bar.setValue(int((vertical_scroll_bar.maximum() - vertical_scroll_bar.minimum()) / 2))

    def wheelEvent(self, event):
        """Handles mouse wheel events for zooming."""
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                zoom_step = 10
                if delta > 0:
                    self.zoom_level = min(self.zoom_level + zoom_step, 400)
                elif delta < 0:
                    self.zoom_level = max(self.zoom_level - zoom_step, 1)
                self.zoom_slider.setValue(self.zoom_level)
                self.update_zoom()
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def open_class_editor(self):
        """Opens the class editor dialog."""
        dialog = ClassEditorDialog(self, self.classes)
        if dialog.exec_() == QDialog.Accepted:
            self.classes = dialog.get_classes()
            self.save_classes()
            self.populate_class_combo()

    def load_classes(self):
        """Loads classes from the classes.yaml file, including colors (DEBUGGING VERSION)."""
        try:
            with open("classes.yaml", "r") as f:
                yaml_data = yaml.safe_load(f)
                print("--- YAML Data Loaded ---")
                print(yaml_data)

                if yaml_data and 'classes' in yaml_data:
                    classes_yaml = yaml_data['classes']
                    print("--- Classes YAML Section ---")
                    print(classes_yaml)
                    self.classes = {}
                    for class_name, class_data in classes_yaml.items():
                        print(f"--- Processing Class: {class_name} ---")
                        print(f"Class Data: {class_data}")
                        color_rgb = class_data.get('color')
                        print(f"Color RGB from YAML: {color_rgb}")
                        if color_rgb and len(color_rgb) == 3:
                            color = QColor(color_rgb[0], color_rgb[1], color_rgb[2])
                            print(f"QColor created: RGB({color.red()}, {color.green()}, {color.blue()})")
                            self.classes[class_name] = color
                        else:
                            print(f"Warning: No valid color found for class '{class_name}' in classes.yaml, using default red.")
                            self.classes[class_name] = QColor(255, 0, 0)
                else:
                    print("Warning: 'classes' section not found or empty in classes.yaml. Using default class (Red).")
                    self.classes = {"Default": QColor(255, 0, 0)}
        except FileNotFoundError:
            self.classes = {"Default": QColor(255, 0, 0)}
        except yaml.YAMLError as e:
            QMessageBox.critical(self, "Error", f"Error parsing classes.yaml: {e}")
            self.classes = {"Default": QColor(255, 0, 0)}
        self.populate_class_combo()

    def save_classes(self):
        """Saves classes to the classes.yaml file in YAML format, including colors (DEBUGGING VERSION)."""
        yaml_data = {'classes': {}}
        for class_name, color in self.classes.items():
            yaml_data['classes'][class_name] = {
                'color': [color.red(), color.green(), color.blue()]
            }

        try:
            with open("classes.yaml", "w") as f:
                yaml.dump(yaml_data, f, indent=2, sort_keys=False)
                print("--- classes.yaml SAVED ---")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save classes to classes.yaml: {e}")

        # IMMEDIATE READ-BACK AND PRINT FOR DEBUGGING
        try:
            with open("classes.yaml", "r") as f:
                saved_yaml_content = f.read()
                print("--- IMMEDIATELY READ BACK classes.yaml ---")
                print(saved_yaml_content)
        except Exception as e:
            print(f"Error reading back classes.yaml after saving: {e}")

    def populate_class_combo(self):
        """Populates the class selection combo box and updates preview."""
        self.class_combo.clear()
        for class_name in self.classes:
            self.class_combo.addItem(class_name)
        if self.classes:
            self.class_combo.setCurrentIndex(0)
            self.current_class = list(self.classes.keys())[0]
            self.update_color_preview()
        else:
            self.current_class = None
            self.update_color_preview()

    def class_selected(self, index):
        """Handles class selection changes."""
        if index >= 0:
            self.current_class = self.class_combo.itemText(index)
            self.update_color_preview()

    def update_color_preview(self):
        """Updates the color preview label with the selected class color."""
        if self.current_class and self.current_class in self.classes:
            color = self.classes[self.current_class]
            pixmap = QPixmap(self.color_preview.size())
            pixmap.fill(color)
            self.color_preview.setPixmap(pixmap)
        else:
            self.color_preview.clear()

    def mouse_press_event(self, event):
        """Handles mouse press events (drawing, selecting, and panning)."""
        print(f"Mouse Press Event - Pos: {event.pos().x()}, {event.pos().y()}")
        if event.button() == Qt.MiddleButton:
            if hasattr(self, 'original_pixmap') and self.original_pixmap:
                self.panning = True
                self.pan_start_point = event.pos()
                self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
                event.accept()
                return

        if event.button() == Qt.LeftButton:
            image_path = self.image_paths[self.current_image_index]
            boxes = self.image_boxes.get(image_path, [])
            self.selected_box = None
            for box in boxes:
                x1, y1, x2, y2, _ = box
                if x1 <= event.pos().x() <= x2 and y1 <= event.pos().y() <= y2:
                    self.selected_box = box
                    self.update_boxes()
                    return

            if self.current_class:
                self.drawing = True
                self.start_point = event.pos()
        super().mousePressEvent(event)

    def mouse_move_event(self, event):
        """Handles mouse move events (drawing and panning)."""
        if self.panning:
            if hasattr(self, 'original_pixmap') and self.original_pixmap:
                delta = event.pos() - self.pan_start_point
                h_scrollbar = self.scroll_area.horizontalScrollBar()
                v_scrollbar = self.scroll_area.verticalScrollBar()

                h_scrollbar.setValue(h_scrollbar.value() - delta.x())
                v_scrollbar.setValue(v_scrollbar.value() - delta.y())
                self.pan_start_point = event.pos()
                event.accept()
                return

        if self.drawing:
            self.current_point = event.pos()
            self.update_boxes()
        super().mouseMoveEvent(event)

    def mouse_release_event(self, event):
        """Handles mouse release events (drawing and panning)."""
        if event.button() == Qt.MiddleButton and self.panning:
            self.panning = False
            self.image_label.setCursor(QCursor(Qt.CrossCursor))
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.drawing and self.current_class:
            self.drawing = False
            end_point = event.pos()
            x1 = min(self.start_point.x(), end_point.x())
            y1 = min(self.start_point.y(), end_point.y())
            x2 = max(self.start_point.x(), end_point.x())
            y2 = max(self.start_point.y(), end_point.y())

            image_path = self.image_paths[self.current_image_index]
            if image_path not in self.image_boxes:
                self.image_boxes[image_path] = []
            self.image_boxes[image_path].append((x1, y1, x2, y2, self.current_class))
            self.update_boxes()
        super().mouseReleaseEvent(event)

    def update_boxes(self):
        """Redraws the image with bounding boxes and class labels."""
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            pixmap = self.original_pixmap.copy()
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            image_path = self.image_paths[self.current_image_index]
            boxes = self.image_boxes.get(image_path, [])

            font = painter.font()
            font.setPointSize(max(8, int(12 * self.zoom_level / 100)))
            painter.setFont(font)

            for x1, y1, x2, y2, class_name in boxes:
                color = self.classes[class_name]
                pen_width = 2
                pen_style = Qt.SolidLine
                if (x1, y1, x2, y2, class_name) == self.selected_box:
                    pen_width = 3
                    pen_style = Qt.DashDotLine
                painter.setPen(QPen(color, pen_width, pen_style))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

                # Draw class label above the box
                text_rect = QRect(x1, y1 - 20, x2 - x1, 20)
                painter.setPen(QPen(color))
                painter.setBrush(color)
                painter.drawRect(QRect(x1-2, y1 - 20 -2, x2 - x1 + 4, 20+4))
                painter.setPen(QPen(Qt.white))
                painter.setBrush(Qt.NoBrush)
                painter.drawText(text_rect, Qt.AlignCenter, class_name)


            if self.drawing:
                x1 = self.start_point.x()
                y1 = self.start_point.y()
                x2 = self.current_point.x()
                y2 = self.current_point.y()
                color = self.classes[self.current_class]
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

            painter.end()
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()

    def browse_for_save_dir(self):
        """Opens a dialog to select the default save directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Default Save Directory")
        if dir_path:
            self.default_save_dir = dir_path
            self.default_save_dir_edit.setText(dir_path)
            self.save_settings()

    def save_annotations(self):
        """Saves annotations for the *current* image."""
        self.save_annotation_for_image(self.image_paths[self.current_image_index])

    def save_all_annotations(self):
        """Saves annotations for *all* images."""
        for image_path in self.image_paths:
            self.save_annotation_for_image(image_path)
        QMessageBox.information(self, "Success", "All annotations saved!")

    def save_annotation_for_image(self, image_path):
        """Saves annotations for a specific image."""
        if self.default_save_dir:
            folder_path = self.default_save_dir
        else:
            folder_path = os.path.dirname(image_path)

        labels_folder = os.path.join(folder_path, "labels")
        os.makedirs(labels_folder, exist_ok=True)

        boxes = self.image_boxes.get(image_path, [])

        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Could not read image for saving: {image_path}")
            return

        height, width, _ = img.shape
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        label_file_path = os.path.join(labels_folder, f"{name}.txt")

        with open(label_file_path, "w+") as f:
            for x1, y1, x2, y2, class_name in boxes:
                class_id = list(self.classes.keys()).index(class_name)

                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    def delete_selected_box(self):
        """Deletes the currently selected bounding box."""
        if self.selected_box:
            image_path = self.image_paths[self.current_image_index]
            boxes = self.image_boxes.get(image_path, [])
            if self.selected_box in boxes:
                boxes.remove(self.selected_box)
                self.image_boxes[image_path] = boxes
                self.selected_box = None
                self.update_boxes()

    def save_settings(self):
        """Saves the current settings (default save directory) to settings.txt."""
        try:
            with open("settings.txt", "w") as f:
                f.write(f"default_save_dir={self.default_save_dir}\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {e}")

    def load_settings(self):
        """Loads settings from settings.txt."""
        try:
            with open("settings.txt", "r") as f:
                for line in f:
                    key, value = line.strip().split("=")
                    if key == "default_save_dir":
                        self.default_save_dir = value
                        self.default_save_dir_edit.setText(value)
        except FileNotFoundError:
            pass

    def browse_export_dir(self):
        """Opens a dialog to select the export directory for the dataset."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if dir_path:
            self.export_dir_edit.setText(dir_path)

    def export_dataset(self):
        """Exports the dataset into train, valid, test folders with labels and images."""
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

        # 1. Validate Percentages (Optional)
        if train_percent + valid_percent + test_percent != 100:
            QMessageBox.warning(self, "Warning", "Train, Valid, Test percentages must sum to 100%.")
            return

        # 2. Create Export Folders
        try:
            os.makedirs(export_dir, exist_ok=True)  # Main export directory
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

        # 3. Split Image Paths
        random.shuffle(self.image_paths)  # Shuffle image paths for random split
        num_images = len(self.image_paths)
        train_split_index = int(num_images * train_percent / 100)
        valid_split_index = train_split_index + int(num_images * valid_percent / 100)

        train_images = self.image_paths[:train_split_index]
        valid_images = self.image_paths[train_split_index:valid_split_index]
        test_images = self.image_paths[valid_split_index:]

        print(
            f"Train images: {len(train_images)}, Valid images: {len(valid_images)}, Test images: {len(test_images)}")  # Debug print

        # 4. Copy Images and Labels
        image_sets = [
            (train_images, train_img_dir, train_label_dir),
            (valid_images, valid_img_dir, valid_label_dir),
            (test_images, test_img_dir, test_label_dir),
        ]

        for image_list, img_dir, label_dir in image_sets:
            for image_path in image_list:
                try:
                    # Copy Image
                    filename = os.path.basename(image_path)
                    dest_image_path = os.path.join(img_dir, filename)
                    shutil.copy2(image_path, dest_image_path)  # copy2 preserves metadata

                    # Copy Label (if annotations exist)
                    if image_path in self.image_boxes:
                        boxes = self.image_boxes[image_path]
                        if boxes:  # Only create label file if boxes exist
                            label_filename = os.path.splitext(filename)[0] + ".txt"
                            dest_label_path = os.path.join(label_dir, label_filename)
                            self.save_annotation_to_path(image_path, dest_label_path,
                                                         boxes)  # Helper function to save labels
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error copying files: {e}")
                    return  # Stop export on error

        # 5. Create data.yaml
        data_yaml_path = os.path.join(export_dir, "data.yaml")
        data_yaml_content = {
            'train': os.path.relpath(train_img_dir, export_dir).replace("\\", "/"),
            # Relative paths, use forward slashes
            'val': os.path.relpath(valid_img_dir, export_dir).replace("\\", "/"),
            'test': os.path.relpath(test_img_dir, export_dir).replace("\\", "/"),
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        try:
            with open(data_yaml_path, 'w') as outfile:
                yaml.dump(data_yaml_content, outfile, default_flow_style=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error writing data.yaml: {e}")
            return

        # 6. Create train_config.yaml
        train_config_path = os.path.join(export_dir, "train_config.yaml")
        train_config_content = {
            'model_weights': self.model_weights_combo.currentText(),
            'data_yaml': os.path.join(".", "data.yaml").replace("\\", "/"),  # Relative path to data.yaml in export dir
            'epochs': self.epochs_spinbox.value(),
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

        # Enable Start Training button after successful export:
        self.start_training_button.setEnabled(True)

    def start_training(self):
        """Starts the YOLOv8 training process in a separate thread/process."""
        export_dir = self.export_dir_edit.text()
        if not export_dir:
            QMessageBox.warning(self, "Warning", "Please select an export directory first.")
            return

        train_config_path = os.path.join(export_dir, "train_config.yaml")
        if not os.path.exists(train_config_path):
            QMessageBox.critical(self, "Error",
                                 f"train_config.yaml not found in export directory: {export_dir}. Please export the dataset first.")
            return

        # Disable Start Training button
        self.start_training_button.setEnabled(False)
        QApplication.processEvents()

        # Start training in a separate process
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.training_finished)

        command = [".venv\\Scripts\\python.exe", "train_script.py", "--config", train_config_path] # Explicit path to venv python (adjust if needed)

        # --- Set Working Directory and Environment ---
        working_directory = os.path.dirname(os.path.abspath(__file__)) # Project script's directory as working dir
        env = QProcessEnvironment.systemEnvironment() # Get system env
        env.insert("PYTHONPATH", ".;.venv\\Lib\\site-packages") # Explicitly add venv site-packages to PYTHONPATH (might be needed)
        self.process.setWorkingDirectory(working_directory) # Set working directory
        self.process.setProcessEnvironment(env) # Set environment for process

        self.process.start(command[0], command[1:])

        QMessageBox.information(self, "Info", "Training started in background. Check console for output.")


    def execute_training(self, config_path): # Not directly used anymore, using subprocess instead
        """Executes the YOLOv8 training process (Direct Ultralytics API - Not used in final version)."""
        try:
            # --- Step 1: Load config from YAML --- (Replicated logic, but better to use train_script.py)
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # --- Step 2: Load the YOLOv8 model ---
            model_weights = config["model_weights"]
            model = YOLO(model_weights)

            # --- Step 3 & 4: Train ---
            model.train(
                data=config["data_yaml"], # Path in train_config.yaml is already relative
                epochs=config["epochs"],
                imgsz=config["imgsz"],
                batch=config["batch_size"],
                lr0=config["lr0"],
                name=config["run_name"],
                device="cuda" if torch.cuda.is_available() else "cpu" # Device from config or detect here
            )

            # --- Step 5: Optionally copy best.pt --- (Replicated logic)
            if config.get("save_best", False):
                run_dir = os.path.join("runs", "detect", config["run_name"], "weights")
                best_pt_src = os.path.join(run_dir, "best.pt")
                best_pt_dest = os.path.join(os.path.dirname(config_path), "best.pt") # Save best.pt in export dir

                if os.path.exists(best_pt_src):
                    shutil.copy2(best_pt_src, best_pt_dest)
                    print(f"Copied best.pt to {best_pt_dest}")
                else:
                    print("best.pt not found! Check if training completed successfully.")

            QMessageBox.information(self, "Success", "Training completed successfully! Check console for details and 'runs' folder for results.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {e}")
        finally:
            self.start_training_button.setEnabled(True) # Re-enable button after training or error


    def handle_stdout(self):
        """Handles standard output from the training process and redirects to QTextEdit."""
        stdout = self.process.readAllStandardOutput().data().decode()
        self.training_console.append(stdout) # Append stdout to QTextEdit


    def handle_stderr(self):
        """Handles standard error output from the training process and redirects to QTextEdit."""
        stderr = self.process.readAllStandardError().data().decode()
        self.training_console.append(f"<span style='color:red;'>{stderr}</span>") # Append stderr in red color to QTextEdit


    def training_finished(self):
        """Handles the completion of the training process."""
        self.start_training_button.setEnabled(True) # Re-enable button
        QMessageBox.information(self, "Info", "Training process finished. Check console and 'runs' folder.")
        self.process = None # Clean up process reference


    def save_annotation_to_path(self, image_path, label_file_path, boxes):
        """Saves annotations to a specified label file path (used for export)."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image to get dimensions: {image_path}")
            return

        height, width, _ = img.shape

        with open(label_file_path, "w+") as f:
            for x1, y1, x2, y2, class_name in boxes:
                class_id = list(self.classes.keys()).index(class_name)

                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


    def browse_export_dir(self):
        """Opens a dialog to select the export directory for the dataset."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if dir_path:
            self.export_dir_edit.setText(dir_path)


# --- ClassEditorDialog and InputDialog remain the same ---
class ClassEditorDialog(QDialog):
    def __init__(self, parent=None, classes=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Classes")
        self.classes = classes or {}

        self.class_list = QListWidget()
        self.class_list.itemDoubleClicked.connect(self.edit_class)

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_class)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_class)
        self.edit_button.setEnabled(False)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_class)
        self.delete_button.setEnabled(False)

        self.accept_button = QPushButton("OK")
        self.accept_button.clicked.connect(self.accept)

        self.reject_button = QPushButton("Cancel")
        self.reject_button.clicked.connect(self.reject)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.delete_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.class_list)
        main_layout.addLayout(button_layout)

        button_box = QHBoxLayout()
        button_box.addWidget(self.accept_button)
        button_box.addWidget(self.reject_button)
        main_layout.addLayout(button_box)

        self.class_list.itemSelectionChanged.connect(self.update_button_states)
        self.populate_list()

    def populate_list(self):
        self.class_list.clear()
        for class_name, color in self.classes.items():
            item = QListWidgetItem(class_name)
            item.setForeground(color)
            self.class_list.addItem(item)

    def add_class(self):
        class_name, ok = InputDialog.getText(self, "New Class", "Enter class name:")
        if ok and class_name:
            if class_name in self.classes:
                QMessageBox.warning(self, "Warning", "Class name already exists.")
                return
            color = QColorDialog.getColor(Qt.red, self, "Choose Class Color")
            if color.isValid():
                self.classes[class_name] = color
                self.populate_list()

    def edit_class(self):
        selected_item = self.class_list.currentItem()
        if selected_item:
            old_class_name = selected_item.text()
            class_name, ok = InputDialog.getText(self, "Edit Class", "Enter new class name:", old_class_name)
            if ok and class_name:
                if class_name != old_class_name and class_name in self.classes:
                    QMessageBox.warning(self, "Warning", "Class name already exists.")
                return
                color = QColorDialog.getColor(self.classes[old_class_name], self, "Choose Class Color")
                if color.isValid():
                    self.classes[class_name] = color
                    if class_name != old_class_name:
                        del self.classes[old_class_name]
                    self.populate_list()

    def delete_class(self):
        selected_item = self.class_list.currentItem()
        if selected_item:
            class_name = selected_item.text()
            reply = QMessageBox.question(self, "Delete Class", f"Are you sure you want to delete '{class_name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.classes[class_name]
                self.populate_list()

    def get_classes(self):
        return self.classes

    def update_button_states(self):
        has_selection = self.class_list.currentItem() is not None
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)


class InputDialog(QDialog):
    """A simple dialog to get text input from the user."""
    def __init__(self, parent=None, title="", label="", text=""):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.label = QLabel(label)
        self.text_edit = QLineEdit(text)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.button_box.addWidget(self.ok_button)
        self.button_box.addWidget(self.cancel_button)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addLayout(self.button_box)

    @staticmethod
    def getText(parent=None, title="", label="", text=""):
        """Static method to create and show the dialog."""
        dialog = InputDialog(parent, title, label, text)
        result = dialog.exec_()
        return dialog.text_edit.text(), result == QDialog.Accepted


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())