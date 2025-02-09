import os
import cv2
import yaml

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog,
    QHBoxLayout, QSlider, QComboBox, QMessageBox, QShortcut,
    QDialog, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsPixmapItem, QGraphicsView
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QColor, QCursor, QKeySequence, QBrush
)
from PyQt5.QtCore import Qt, QRectF, QPointF

from class_editor import ClassEditorDialog, InputDialog
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsTextItem
from PyQt5.QtGui import QPen, QBrush, QColor, QFont
from PyQt5.QtCore import Qt, QRectF


class BoundingBoxItem(QGraphicsRectItem):
    """
    A custom QGraphicsRectItem subclass that displays a rectangular bounding box
    plus a text label for the associated class name. Used for drawing annotations.

    Things you might want to edit/adjust in the future:
      - The alpha (transparency) of the fill brush
      - The outline thickness or color
      - The default font size and color for the label
      - The logic in itemChange if you want bounding boxes to do more on move/resize
      - The label placement if you want it centered or offset
    """
    def __init__(self, rect, class_name, color, annotation_tab=None):
        """
        :param rect:       A QRectF indicating the initial position/size of this box.
        :param class_name: A string name for the bounding box class (e.g. "Person").
        :param color:      A QColor or similar indicating the box color.
        :param annotation_tab: (Optional) reference back to the AnnotationTab or parent.
                              Helps if you need to notify it when boxes move or get deleted.
        """
        super().__init__(rect)

        self.class_name = class_name
        self.color = color
        self.annotation_tab = annotation_tab  # If needed, for callbacks or data updates

        # 1) Configure bounding-box appearance
        #    - The outline is drawn with a QPen. Here, 2 px wide.
        self.setPen(QPen(self.color, 2))

        #    - The fill uses a brush with semi-transparency (alpha=80).
        #      Increase alpha to make it more opaque, decrease for more see-through.
        self.setBrush(QBrush(QColor(self.color.red(),
                                    self.color.green(),
                                    self.color.blue(),
                                    80)))  # Adjust '80' to your liking (0..255)

        # 2) Make the item selectable, movable, and track changes.
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)

        # 3) Add a text label to show the class name
        #    - QGraphicsTextItem is a child of this rect item, so it moves with us
        self.label_text = QGraphicsTextItem(self.class_name, self)

        #    - Set a font if you want it bigger or bolder
        font = QFont("Arial", 20, QFont.Bold)
        self.label_text.setFont(font)

        #    - The default text color is white; change if your fill is too light
        self.label_text.setDefaultTextColor(Qt.white)

        #    - Position it at the top-left corner of the bounding box
        self.label_text.setPos(rect.topLeft())

    def itemChange(self, change, value):
        """
        Called whenever the item moves, is reparented, or changes geometry.
        If you need to notify your annotation_tab or do something special
        when the box is moved, you can do it here.
        """
        if change == QGraphicsRectItem.ItemPositionChange:
            # This event fires when the user drags/moves the box
            new_rect = self.rect().translated(value - self.pos())
            # Move the label to match our new top-left
            self.label_text.setPos(new_rect.topLeft())

        elif change == QGraphicsRectItem.ItemPositionHasChanged:
            # After the box has finished moving, you could call back to the parent
            # e.g. self.annotation_tab.update_image_info() if you want live updates.
            if self.annotation_tab:
                self.annotation_tab.update_image_info()

        # If the item is removed from the scene, you can remove it from the parent's data
        # For example, if change == QGraphicsRectItem.ItemSceneChange and value is None:
        #    # The item is being removed from the scene
        #    if self.annotation_tab:
        #       # Purge from self.annotation_tab.image_boxes, etc.

        return super().itemChange(change, value)

    def setRect(self, rect):
        """
        Override setRect to keep the label pinned to the top-left corner.
        Called whenever you change the bounding rectangle (like while drawing).
        """
        super().setRect(rect)
        self.label_text.setPos(self.rect().topLeft())


class CustomGraphicsView(QGraphicsView):
    """
    A subclass of QGraphicsView that handles:
      - Left-click-drag for drawing bounding boxes,
      - Middle-click-drag for panning (wheel click),
      - Cursor is a crosshair by default, changes to closed-hand while panning,
      - No extra guide lines on the screen.
    """

    def __init__(self, annotation_tab, parent=None):
        super().__init__(parent)
        self.annotation_tab = annotation_tab
        self.setDragMode(QGraphicsView.NoDrag)
        # By default, show a crosshair
        self.setCursor(Qt.CrossCursor)

        # Track bounding-box drawing
        self.drawing = False
        self.start_point = None
        self.current_rect_item = None

    def mousePressEvent(self, event):
        # MIDDLE button => panning
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ClosedHandCursor)   # closed-hand while panning
            self.setInteractive(False)
            super().mousePressEvent(event)
            return

        # LEFT button => draw bounding box if a class is selected
        if event.button() == Qt.LeftButton and self.annotation_tab.current_class:
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())

            # Import your bounding-box item class if needed:
            from annotation_tab import BoundingBoxItem
            color = self.annotation_tab.classes[self.annotation_tab.current_class]

            rect_item = BoundingBoxItem(
                QRectF(self.start_point, self.start_point),
                self.annotation_tab.current_class,
                color,
                annotation_tab=self.annotation_tab
            )
            self.annotation_tab.scene.addItem(rect_item)
            self.current_rect_item = rect_item
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # If we're drawing a box, update its rectangle
        if self.drawing and self.current_rect_item:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()
            self.current_rect_item.setRect(rect)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # If panning with middle button
        if event.button() == Qt.MiddleButton:
            # Go back to crosshair cursor
            self.setCursor(Qt.CrossCursor)
            self.setDragMode(QGraphicsView.NoDrag)
            self.setInteractive(True)
            super().mouseReleaseEvent(event)
            return

        # If drawing a bounding box with left button
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()

            # If the box is tiny, remove it
            if rect.width() < 5 or rect.height() < 5:
                if self.current_rect_item:
                    self.annotation_tab.scene.removeItem(self.current_rect_item)
            else:
                # Store it in annotation_tab’s data structure
                image_path = self.annotation_tab.image_paths[self.annotation_tab.current_image_index]
                if image_path not in self.annotation_tab.image_boxes:
                    self.annotation_tab.image_boxes[image_path] = []
                self.annotation_tab.image_boxes[image_path].append(
                    (rect, self.annotation_tab.current_class)
                )

            self.current_rect_item = None
            return

        super().mouseReleaseEvent(event)

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

        self.zoom_slider = QSlider(Qt.Horizontal, self)
        self.zoom_slider.setRange(1, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom_slider)
        self.zoom_label = QLabel("Zoom: 100%", self)

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

        zoom_hbox = QHBoxLayout()
        zoom_hbox.addWidget(QLabel("Zoom:"))
        zoom_hbox.addWidget(self.zoom_slider)
        zoom_hbox.addWidget(self.zoom_label)

        class_hbox = QHBoxLayout()
        class_hbox.addWidget(QLabel("Class:"))
        class_hbox.addWidget(self.class_combo)
        class_hbox.addWidget(self.color_preview)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_hbox)
        main_layout.addLayout(nav_hbox)

        # Instead of a normal QGraphicsView, we use our custom subclass
        self.scene = QGraphicsScene(self)
        self.image_view = CustomGraphicsView(annotation_tab=self)  # pass self so it can call back
        self.image_view.setScene(self.scene)

        main_layout.addWidget(self.image_view)
        main_layout.addLayout(zoom_hbox)
        main_layout.addWidget(self.edit_classes_button)
        main_layout.addLayout(class_hbox)
        main_layout.addWidget(self.save_button)
        main_layout.addWidget(self.save_all_button)

        # Data
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

        # Load classes, settings
        self.load_classes()
        self.load_settings()
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

                # scene rect
                self.scene.setSceneRect(QRectF(pixmap.rect()))

                # Fit in view
                self.image_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                self.image_view.viewport().setCursor(QCursor(Qt.CrossCursor))

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

    # ---------------- ZOOMING -----------------

    def update_zoom_slider(self):
        zoom_level = self.zoom_slider.value()
        self.zoom_label.setText(f"Zoom: {zoom_level}%")
        self.image_view.resetTransform()
        factor = zoom_level / 100.0
        self.image_view.scale(factor, factor)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            zoom_factor = 1.15
            if event.angleDelta().y() > 0:
                self.image_view.scale(zoom_factor, zoom_factor)
            else:
                self.image_view.scale(1 / zoom_factor, 1 / zoom_factor)
            self.update_zoom_from_view()
            event.accept()
        else:
            super().wheelEvent(event)

    def update_zoom_from_view(self):
        transform = self.image_view.transform()
        current_scale = transform.m11()
        zoom_percentage = int(current_scale * 100)
        self.zoom_slider.setValue(zoom_percentage)
        self.zoom_label.setText(f"Zoom: {zoom_percentage}%")

    # ---------------- BOUNDING BOX ANNOTATIONS -----------------

    def load_annotations(self):
        if not self.image_paths or not self.pixmap_item:
            return
        # remove any existing boxes from scene
        for it in self.scene.items():
            if isinstance(it, BoundingBoxItem):
                self.scene.removeItem(it)

        image_path = self.image_paths[self.current_image_index]
        self.image_boxes[image_path] = []  # reset in memory

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
        for it in self.scene.selectedItems():
            if isinstance(it, BoundingBoxItem):
                # remove from data
                for bd in self.image_boxes[image_path]:
                    if bd[0] == it.rect():
                        self.image_boxes[image_path].remove(bd)
                        break
                self.scene.removeItem(it)

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

        self.populate_class_combo()

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
