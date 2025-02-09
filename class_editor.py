# class_editor.py
from PyQt5.QtWidgets import (QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QLineEdit, QColorDialog,
                             QMessageBox)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

class ClassEditorDialog(QDialog):
    """Dialog for editing class names and colors."""
    def __init__(self, parent=None, classes=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Classes")
        self.classes = classes or {}  # Use existing classes or an empty dictionary

        self.class_list = QListWidget()
        self.class_list.itemDoubleClicked.connect(self.edit_class)  # Double-click to edit

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_class)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_class)
        self.edit_button.setEnabled(False)  # Disable initially

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_class)
        self.delete_button.setEnabled(False)  # Disable initially

        self.accept_button = QPushButton("OK")
        self.accept_button.clicked.connect(self.accept)

        self.reject_button = QPushButton("Cancel")
        self.reject_button.clicked.connect(self.reject)

        # --- Layout ---
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

        self.class_list.itemSelectionChanged.connect(self.update_button_states)  # Enable/disable buttons
        self.populate_list()  # Populate the list with initial classes

    def populate_list(self):
        """Populates the list widget with class names and colors."""
        self.class_list.clear()
        for class_name, color in self.classes.items():
            item = QListWidgetItem(class_name)
            item.setForeground(color)  # Set text color
            self.class_list.addItem(item)

    def add_class(self):
        """Adds a new class."""
        class_name, ok = InputDialog.getText(self, "New Class", "Enter class name:")
        if ok and class_name:
            if class_name in self.classes:
                QMessageBox.warning(self, "Warning", "Class name already exists.")
                return
            color = QColorDialog.getColor(Qt.red, self, "Choose Class Color")  # Default to red
            if color.isValid():
                self.classes[class_name] = color
                self.populate_list()

    def edit_class(self):
        """Edits an existing class."""
        selected_item = self.class_list.currentItem()
        if selected_item:
            old_class_name = selected_item.text()
            class_name, ok = InputDialog.getText(self, "Edit Class", "Enter new class name:", text=old_class_name)
            if ok and class_name:
                if class_name != old_class_name and class_name in self.classes:
                    QMessageBox.warning(self, "Warning", "Class name already exists.")
                    return
                color = QColorDialog.getColor(self.classes[old_class_name], self, "Choose Class Color")
                if color.isValid():
                    # Update the class name and color
                    self.classes[class_name] = color
                    if class_name != old_class_name:
                        del self.classes[old_class_name]  # Remove old entry
                    self.populate_list()

    def delete_class(self):
        """Deletes the selected class."""
        selected_item = self.class_list.currentItem()
        if selected_item:
            class_name = selected_item.text()
            reply = QMessageBox.question(self, "Delete Class", f"Are you sure you want to delete '{class_name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.classes[class_name]
                self.populate_list()

    def get_classes(self):
        """Returns the updated dictionary of classes."""
        return self.classes

    def update_button_states(self):
        """Enables/disables the Edit and Delete buttons based on selection."""
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