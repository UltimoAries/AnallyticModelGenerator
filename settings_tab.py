# settings_tab.py
import os
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, QFormLayout,
                             QFileDialog, QMessageBox)  # Import only necessary components
from PyQt5.QtCore import pyqtSignal, QObject


class SettingsTab(QWidget):

    settings_changed = pyqtSignal()  # Custom signal for when settings change

    def __init__(self):
        super().__init__()

        self.default_save_dir_label = QLabel("Default Save Directory:", self)
        self.default_save_dir_edit = QLineEdit(self)
        self.browse_button = QPushButton("Browse...", self)
        self.browse_button.clicked.connect(self.browse_for_save_dir)

        # --- Layout ---
        layout = QFormLayout(self)
        layout.addRow(self.default_save_dir_label, self.default_save_dir_edit)
        layout.addRow(self.browse_button)

        self.load_settings() # Load settings on creation

    def browse_for_save_dir(self):
        """Opens a dialog to select the default save directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Default Save Directory")
        if dir_path:
            self.default_save_dir_edit.setText(dir_path)
            self.save_settings()  # Save immediately on change.
            self.settings_changed.emit()  # Emit the signal

    def save_settings(self):
        """Saves settings (like default save directory) to a file."""
        try:
            with open("settings.txt", "w") as f:
                f.write(f"default_save_dir={self.default_save_dir_edit.text()}\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {e}")

    def load_settings(self):
        """Loads settings from a file."""
        try:
            with open("settings.txt", "r") as f:
                for line in f:
                    key, value = line.strip().split("=")
                    if key == "default_save_dir":
                        self.default_save_dir_edit.setText(value)
        except FileNotFoundError:
            pass  # It's okay if the settings file doesn't exist yet

    def get_default_save_dir(self):  # Added a getter method
        """Returns the current default save directory."""
        return self.default_save_dir_edit.text()