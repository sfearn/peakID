from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QLineEdit, QLabel, QMessageBox
)
from PyQt5.QtGui import QPalette, QColor
import sys


class FileEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.default_file = 'config.py'
        self.selected_file = None  # To store the chosen spectrum file

        # Initialize UI elements
        self.initUI()

        # Load default file content if it exists
        content = self.open_file(self.default_file)
        if content is not None:
            self.editor.setPlainText(content)

    def initUI(self):
        # Set up main window
        self.setWindowTitle("File Editor")
        self.setGeometry(100, 100, 600, 400)
        self.set_dark_mode()  # Set dark mode

        # Editor area
        self.editor = QTextEdit(self)
        self.editor.setStyleSheet("background-color: #333333; color: #ffffff;")

        # Spectrum file input
        self.spectrum_file_input = QLineEdit(self)
        self.spectrum_file_input.setStyleSheet("background-color: #333333; color: #ffffff;")

        # Browse button with dark blue color
        self.spectrum_file_button = QPushButton("Browse Spectrum File", self)
        self.spectrum_file_button.setStyleSheet("background-color: #3b5998; color: #ffffff;")
        self.spectrum_file_button.clicked.connect(self.browse_file)

        # Save button with dark blue color
        self.save_button = QPushButton("Save", self)
        self.save_button.setStyleSheet("background-color: #3b5998; color: #ffffff;")
        self.save_button.clicked.connect(self.save_file)

        # Run Peak Finder button with dark blue color
        self.run_peak_finder_button = QPushButton("Run Peak Finder", self)
        self.run_peak_finder_button.setStyleSheet("background-color: #3b5998; color: #ffffff;")
        self.run_peak_finder_button.clicked.connect(self.run_peak_finder)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Configuration File Editor:"))
        layout.addWidget(self.editor)
        layout.addWidget(self.save_button)  # Save button placed above spectrum file input
        layout.addWidget(QLabel("Enter file path for file containing spectrum:"))
        layout.addWidget(self.spectrum_file_input)
        layout.addWidget(self.spectrum_file_button)
        layout.addWidget(self.run_peak_finder_button)

        # Container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def set_dark_mode(self):
        """Set up a dark mode color scheme."""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

        app.setPalette(dark_palette)

    def show_message(self, title, message, is_error=False):
        """Show a message box in dark mode."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStyleSheet(
            "QMessageBox { background-color: #333333; color: #ffffff; }"
            "QPushButton { background-color: #3b5998; color: #ffffff; }"
        )
        if is_error:
            msg_box.setIcon(QMessageBox.Critical)
        else:
            msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

    def open_file(self, filename):
        try:
            with open(filename, 'r') as file:
                return file.read()
        except FileNotFoundError:
            self.show_message("Error", f"File '{filename}' not found.", is_error=True)
        except Exception as e:
            self.show_message("Error", f"An error occurred while opening the file:\n{e}", is_error=True)
        return None

    def save_file(self):
        filename = self.default_file
        content = self.editor.toPlainText()
        try:
            with open(filename, 'w') as file:
                file.write(content)
            self.show_message("Success", f"File '{filename}' saved successfully.")
        except Exception as e:
            self.show_message("Error", f"An error occurred while saving the file:\n{e}", is_error=True)

    def browse_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Spectrum File", "", "All Files (*)", options=options)
        if file:
            self.spectrum_file_input.setText(file)

    def run_peak_finder(self):
        spectrum_file = self.spectrum_file_input.text()
        if spectrum_file:
            self.selected_file = spectrum_file  # Store selected file path
            self.close()  # Close the window
        else:
            self.show_message("Warning", "Please select a spectrum file before running Peak Finder.", is_error=True)


def run_GUI():
    global app
    app = QApplication(sys.argv)
    editor_app = FileEditorApp()
    editor_app.show()
    app.exec_()

    # Access the selected spectrum file path
    if editor_app.selected_file:
        print("Selected Spectrum File:", editor_app.selected_file)
    else:
        print("No spectrum file was selected.")

    return editor_app.selected_file


if __name__ == "__main__":
    run_GUI()
    print("GUI closed")
