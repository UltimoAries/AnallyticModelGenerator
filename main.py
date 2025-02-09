# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from annotation_tab import AnnotationTab
from settings_tab import SettingsTab
from training_tab import TrainingTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool")
        self.setGeometry(100, 100, 800, 600)

        self.tab_widget = QTabWidget()
        self.annotation_tab = AnnotationTab()
        self.settings_tab = SettingsTab()
        self.training_tab = TrainingTab()

        self.tab_widget.addTab(self.annotation_tab, "Annotation")
        self.tab_widget.addTab(self.training_tab, "Training")
        self.tab_widget.addTab(self.settings_tab, "Settings")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tab_widget)
        self.setCentralWidget(central_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
