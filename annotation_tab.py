import os
import cv2
import yaml

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QHBoxLayout, QComboBox, QMessageBox, QShortcut,
    QDialog, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsPixmapItem, QGraphicsView
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QColor, QCursor, QKeySequence, QBrush, QFont
)
from PyQt5.QtCore import Qt, QRectF

from class_editor import ClassEditorDialog, InputDialog
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTransform


class BoundingBoxItem(QGraphicsRectItem):
    """
    A custom QGraphicsRectItem subclass that displays a rectangular bounding box
    plus a text label for the associated class name.
    """
    def __init__(self, rect, class_name, color, annotation_tab=None):
        super().__init__(rect)
        self.class_name = class_name
        self.color = color
        self.annotation_tab = annotation_tab

        # Configure appearance
        self.setPen(QPen(self.color, 2))
        self.setBrush(QBrush(QColor(self.color.red(),
                                    self.color.green(),
                                    self.color.blue(),
                                    80)))
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)

        # Add a text label
        self.label_text = QGraphicsTextItem(self.class_name, self)
        font = QFont("Arial", 20, QFont.Bold)
        self.label_text.setFont(font)
        self.label_text.setDefaultTextColor(Qt.white)
        self.label_text.setPos(rect.topLeft())

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange:
            new_rect = self.rect().translated(value - self.pos())
            self.label_text.setPos(new_rect.topLeft())
        elif change == QGraphicsRectItem.ItemPositionHasChanged:
            if self.annotation_tab:
                self.annotation_tab.update_image_info()
        return super().itemChange(change, value)

    def setRect(self, rect):
        super().setRect(rect)
        self.label_text.setPos(self.rect().topLeft())


class CustomGraphicsView(QGraphicsView):
    def __init__(self, annotation_tab, parent=None):
        super().__init__(parent)
        self.annotation_tab = annotation_tab
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.CrossCursor)

        # Set the transformation anchor so that zooming is under the mouse.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # For mouse wheel zooming (absolute zoom)
        self.current_zoom = 100  # Zoom percentage

        # For right-click panning: store the start point (if panning)
        self._pan_start = None

        # Variables for left-click drawing (unchanged)
        self.drawing = False
        self.start_point = None
        self.current_rect_item = None

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            # Begin panning: record the starting mouse position.
            self._pan_start = event.pos()
            # Change the cursor to indicate panning.
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.annotation_tab.current_class:
            # Start drawing a bounding box.
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            color = self.annotation_tab.classes[self.annotation_tab.current_class]
            from annotation_tab import BoundingBoxItem  # if needed
            rect_item = BoundingBoxItem(
                QRectF(self.start_point, self.start_point),
                self.annotation_tab.current_class,
                color,
                annotation_tab=self.annotation_tab
            )
            self.annotation_tab.scene.addItem(rect_item)
            self.current_rect_item = rect_item
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Handle panning if right button is held.
        if self._pan_start is not None:
            # Calculate the difference between the current and the starting mouse position.
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()  # update the starting point for the next move

            # Adjust the scroll bars by the negative delta so that the view pans accordingly.
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Handle drawing if left button is active.
        if self.drawing and self.current_rect_item:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()
            self.current_rect_item.setRect(rect)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # End panning on right mouse release.
        if event.button() == Qt.RightButton and self._pan_start is not None:
            self._pan_start = None
            # Restore the cursor to the cross cursor (or any default you prefer).
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return

        # End drawing on left mouse release.
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()

            if rect.width() < 5 or rect.height() < 5:
                if self.current_rect_item:
                    self.annotation_tab.scene.removeItem(self.current_rect_item)
            else:
                image_path = self.annotation_tab.image_paths[self.annotation_tab.current_image_index]
                if image_path not in self.annotation_tab.image_boxes:
                    self.annotation_tab.image_boxes[image_path] = []
                self.annotation_tab.image_boxes[image_path].append(
                    (rect, self.annotation_tab.current_class)
                )
            self.current_rect_item = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Only mouse wheel zooming is enabled (absolute zoom).
        zoom_step = 10  # zoom percentage step per notch

        if event.angleDelta().y() > 0:
            self.current_zoom += zoom_step
        else:
            self.current_zoom -= zoom_step

        # Clamp the zoom level between 10% and 400%.
        self.current_zoom = max(10, min(self.current_zoom, 400))
        factor = self.current_zoom / 100.0

        # Reset and apply the new transform.
        self.setTransform(QTransform().scale(factor, factor))
        event.accept()


class AnnotationTab(QWidget):
    def __init__(self):
        super().__init__()

        # UI elements
        self.load_folder_button = QPushButton("Load Folder", self)
        self.load_folder_button.clicked.connect(self.load_folder)

        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)

        self.image_info_label = QLabel("No folder loaded", self)

        # -- Zooming UI elements have been removed --
        # self.zoom_slider = QSlider(Qt.Horizontal, self)
        # self.zoom_slider.setRange(1, 400)
        # self.zoom_slider.setValue(100)
        # self.zoom_slider.valueChanged.connect(self.update_zoom_slider)
        # self.zoom_label = QLabel("Zoom: 100%", self)

        self.edit_classes_button = QPushButton("Edit Classes", self)
        self.edit_classes_button.clicked.connect(self.open_class_editor)

        self.class_combo = QComboBox(self)
        self.class_combo.currentIndexChanged.connect(self.class_selected)

        self.color_preview = QLabel(self)
        self.color_preview.setFixedSize(20, 20)

        self.save_button = QPushButton("Save Annotations", self)
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)

        self.save_all_button = QPushButton("Save All Annotations", self)
        self.save_all_button.clicked.connect(self.save_all_annotations)
        self.save_all_button.setEnabled(False)

        # Layout
        top_hbox = QHBoxLayout()
        top_hbox.addWidget(self.load_folder_button)
        top_hbox.addWidget(self.image_info_label)
        top_hbox.addStretch(1)

        nav_hbox = QHBoxLayout()
        nav_hbox.addWidget(self.prev_button)
        nav_hbox.addWidget(self.next_button)

        # -- Zoom layout removed --
        # zoom_hbox = QHBoxLayout()
        # zoom_hbox.addWidget(QLabel("Zoom:"))
        # zoom_hbox.addWidget(self.zoom_slider)
        # zoom_hbox.addWidget(self.zoom_label)

        class_hbox = QHBoxLayout()
        class_hbox.addWidget(QLabel("Class:"))
        class_hbox.addWidget(self.class_combo)
        class_hbox.addWidget(self.color_preview)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_hbox)
        main_layout.addLayout(nav_hbox)

        # Set up the graphics scene and view
        self.scene = QGraphicsScene(self)
        self.image_view = CustomGraphicsView(annotation_tab=self)
        self.image_view.setScene(self.scene)

        main_layout.addWidget(self.image_view)
        # main_layout.addLayout(zoom_hbox)  <-- Removed zoom layout
        main_layout.addWidget(self.edit_classes_button)
        main_layout.addLayout(class_hbox)
        main_layout.addWidget(self.save_button)
        main_layout.addWidget(self.save_all_button)

        # Data storage
        self.image_paths = []
        self.current_image_index = -1
        self.classes = {}
        self.image_boxes = {}
        self.pixmap_item = None
        self.current_class = None

        # Shortcuts
        self.prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_shortcut.activated.connect(self.prev_image)
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_shortcut.activated.connect(self.next_image)
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.delete_shortcut.activated.connect(self.delete_selected_box)

        # Load settings and classes
        self.load_classes()
        self.load_settings()
        self.populate_class_combo()
        self.update_color_preview()

    # ---------------- FOLDER / IMAGE LOADING -----------------

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.image_paths = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
            ]
            if self.image_paths:
                self.current_image_index = 0
                self.load_image()
                self.update_image_info()
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                self.save_button.setEnabled(True)
                self.save_all_button.setEnabled(True)
            else:
                self.clear_image()
                self.image_info_label.setText("No images found in folder.")
                self.current_image_index = -1
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
                self.save_button.setEnabled(False)
                self.save_all_button.setEnabled(False)

    def load_image(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            try:
                q_image = QImage(image_path)
                if q_image.isNull():
                    raise ValueError(f"Could not load image at {image_path}")

                pixmap = QPixmap.fromImage(q_image)
                if pixmap.isNull():
                    raise ValueError(f"Could not convert QImage at {image_path}")

                # Remove old pixmap_item if needed
                if self.pixmap_item:
                    self.scene.removeItem(self.pixmap_item)
                    self.pixmap_item = None

                self.pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)

                # Set scene rect
                self.scene.setSceneRect(QRectF(pixmap.rect()))

                # Fit the image in view
                self.image_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                self.image_view.viewport().setCursor(QCursor(Qt.CrossCursor))

                # **Update current_zoom based on the applied fitInView transform**
                # Assuming uniform scaling, m11() gives the horizontal scale factor.
                self.image_view.current_zoom = self.image_view.transform().m11() * 100

                self.load_annotations()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {e}")
                self.clear_image()
        else:
            self.clear_image()

    def clear_image(self):
        self.scene.clear()
        self.pixmap_item = None
        if 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            if image_path in self.image_boxes:
                self.image_boxes[image_path].clear()
        else:
            self.image_boxes.clear()

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image()
            self.update_image_info()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
            self.update_image_info()

    def update_image_info(self):
        if self.image_paths and 0 <= self.current_image_index < len(self.image_paths):
            fname = os.path.basename(self.image_paths[self.current_image_index])
            idx = self.current_image_index + 1
            tot = len(self.image_paths)
            self.image_info_label.setText(f"Image: {fname} ({idx}/{tot})")
        else:
            self.image_info_label.setText("No folder loaded")

    # ---------------- BOUNDING BOX ANNOTATIONS -----------------

    def load_annotations(self):
        if not self.image_paths or not self.pixmap_item:
            return

        # Remove existing bounding boxes from the scene
        for item in self.scene.items():
            if isinstance(item, BoundingBoxItem):
                self.scene.removeItem(item)

        image_path = self.image_paths[self.current_image_index]
        self.image_boxes[image_path] = []

        label_path = os.path.join(
            os.path.dirname(image_path),
            "labels",
            os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        )
        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    for line in f:
                        class_id, x_c, y_c, w, h = map(float, line.strip().split())
                        class_name = list(self.classes.keys())[int(class_id)]
                        color = self.classes[class_name]

                        img_w = self.pixmap_item.pixmap().width()
                        img_h = self.pixmap_item.pixmap().height()

                        x = (x_c - w / 2) * img_w
                        y = (y_c - h / 2) * img_h
                        bw = w * img_w
                        bh = h * img_h

                        rect = QRectF(x, y, bw, bh)
                        box_item = BoundingBoxItem(rect, class_name, color, annotation_tab=self)
                        self.scene.addItem(box_item)
                        self.image_boxes[image_path].append((rect, class_name))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading annotations: {e}")

    def save_annotations(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            self.save_annotation_for_image(self.image_paths[self.current_image_index])
            QMessageBox.information(self, "Saved", "Annotations saved for current image.")

    def save_all_annotations(self):
        for image_path in self.image_paths:
            self.save_annotation_for_image(image_path)
        QMessageBox.information(self, "Saved", "All annotations saved for all images.")

    def save_annotation_for_image(self, image_path):
        if not self.pixmap_item:
            return
        folder_path = os.path.dirname(image_path)
        labels_dir = os.path.join(folder_path, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(labels_dir, f"{name}.txt")

        boxes = self.image_boxes.get(image_path, [])
        img_w = self.pixmap_item.pixmap().width() if self.pixmap_item else 1
        img_h = self.pixmap_item.pixmap().height() if self.pixmap_item else 1

        with open(txt_path, "w") as f:
            for rect, class_name in boxes:
                x_c = rect.center().x() / img_w
                y_c = rect.center().y() / img_h
                w = rect.width() / img_w
                h = rect.height() / img_h
                class_id = list(self.classes.keys()).index(class_name)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    def delete_selected_box(self):
        if self.current_image_index < 0:
            return
        image_path = self.image_paths[self.current_image_index]
        for item in self.scene.selectedItems():
            if isinstance(item, BoundingBoxItem):
                for bd in self.image_boxes[image_path]:
                    if bd[0] == item.rect():
                        self.image_boxes[image_path].remove(bd)
                        break
                self.scene.removeItem(item)

    # ---------------- CLASSES / SETTINGS -----------------

    def load_settings(self):
        try:
            with open("settings.txt", "r") as f:
                for line in f:
                    key, value = line.strip().split("=")
                    if key == "default_save_dir":
                        self.default_save_dir = value
        except FileNotFoundError:
            pass

    def open_class_editor(self):
        dialog = ClassEditorDialog(self, self.classes)
        if dialog.exec_() == QDialog.Accepted:
            self.classes = dialog.get_classes()
            self.save_classes()
            self.populate_class_combo()

    def load_classes(self):
        try:
            with open("classes.yaml", "r") as f:
                data = yaml.safe_load(f)
                if data and "classes" in data:
                    self.classes = {}
                    for cname, cinfo in data["classes"].items():
                        rgb = cinfo.get("color", [255, 0, 0])
                        self.classes[cname] = QColor(*rgb)
                else:
                    self.classes = {"Default": QColor(255, 0, 0)}
        except FileNotFoundError:
            self.classes = {"Default": QColor(255, 0, 0)}
        except yaml.YAMLError as e:
            print(f"YAML error: {e}")
            self.classes = {"Default": QColor(255, 0, 0)}

    def save_classes(self):
        try:
            data = {"classes": {}}
            for cname, color in self.classes.items():
                data["classes"][cname] = {"color": [color.red(), color.green(), color.blue()]}
            with open("classes.yaml", "w") as f:
                yaml.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving classes: {e}")

    def populate_class_combo(self):
        self.class_combo.clear()
        for cname in self.classes:
            self.class_combo.addItem(cname)
        if self.classes:
            self.class_combo.setCurrentIndex(0)
            self.current_class = self.class_combo.itemText(0)
        else:
            self.current_class = None
        self.update_color_preview()

    def class_selected(self, idx):
        if idx >= 0:
            self.current_class = self.class_combo.itemText(idx)
            self.update_color_preview()

    def update_color_preview(self):
        if self.current_class in self.classes:
            color = self.classes[self.current_class]
            pm = QPixmap(self.color_preview.size())
            pm.fill(color)
            self.color_preview.setPixmap(pm)
        else:
            self.color_preview.clear()
