import sys
import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import main as peak_main
from settings_manager import DEFAULT_SETTINGS, load_settings, reset_to_defaults, save_settings


class PeakIDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        peak_main.update_settings(self.settings)
        self.raw_spectrum = None
        self.selected_file = None

        self.setting_widgets = {}
        self.run_option_widgets = {}

        self._build_ui()
        self._load_settings_into_widgets(self.settings)

    def _build_ui(self):
        self.setWindowTitle("PeakID")
        self.resize(1500, 920)

        splitter = QSplitter()

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        file_group = QGroupBox("Spectrum")
        file_layout = QGridLayout(file_group)
        self.file_input = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_file)
        run_button = QPushButton("Run")
        run_button.clicked.connect(self._run_peakid)

        file_layout.addWidget(QLabel("File"), 0, 0)
        file_layout.addWidget(self.file_input, 0, 1)
        file_layout.addWidget(browse_button, 0, 2)
        file_layout.addWidget(run_button, 1, 2)

        controls_layout.addWidget(file_group)

        self._add_settings_groups(controls_layout)

        action_row = QHBoxLayout()
        apply_button = QPushButton("Apply Settings")
        apply_button.clicked.connect(self._apply_settings)
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        action_row.addWidget(apply_button)
        action_row.addWidget(reset_button)
        controls_layout.addLayout(action_row)

        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        right_layout.addWidget(self.canvas)

        right_layout.addWidget(QLabel("Results"))
        self.results_table = QTableWidget()
        right_layout.addWidget(self.results_table)

        right_layout.addWidget(QLabel("Status"))
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(110)
        right_layout.addWidget(self.status_text)

        splitter.addWidget(scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([480, 1020])

        self.setCentralWidget(splitter)

    def _add_settings_groups(self, parent_layout):
        calibration_group = QGroupBox("Calibration")
        calibration_form = QFormLayout(calibration_group)
        self.setting_widgets["calibration_model"] = self._combo(["linear", "parabolic", "none"])
        self.setting_widgets["M"] = QLineEdit()
        self.setting_widgets["c"] = QLineEdit()
        self.setting_widgets["a"] = QLineEdit()
        self.setting_widgets["b"] = QLineEdit()
        calibration_form.addRow("Model", self.setting_widgets["calibration_model"])
        calibration_form.addRow("M", self.setting_widgets["M"])
        calibration_form.addRow("c", self.setting_widgets["c"])
        calibration_form.addRow("a", self.setting_widgets["a"])
        calibration_form.addRow("b", self.setting_widgets["b"])
        parent_layout.addWidget(calibration_group)

        peak_group = QGroupBox("Peak Finding")
        peak_form = QFormLayout(peak_group)
        for key in ["min_FWHM", "max_FWHM", "FWHM_step", "confidence", "intensity_threshold", "LLD", "HLD"]:
            self.setting_widgets[key] = QLineEdit()
            peak_form.addRow(key, self.setting_widgets[key])
        parent_layout.addWidget(peak_group)

        fit_group = QGroupBox("Fitting and Smoothing")
        fit_form = QFormLayout(fit_group)
        for key in ["num_iters", "base_width", "smooth_window", "smooth_order"]:
            self.setting_widgets[key] = QLineEdit()
            fit_form.addRow(key, self.setting_widgets[key])
        parent_layout.addWidget(fit_group)

        background_group = QGroupBox("Background")
        background_form = QFormLayout(background_group)
        self.setting_widgets["background_subtraction"] = self._combo(["none", "rolling_ball", "linear"])
        self.setting_widgets["ballradius"] = QLineEdit()
        self.setting_widgets["gradient"] = QLineEdit()
        self.setting_widgets["gradtype"] = self._combo(["sqrt", "linear"])
        background_form.addRow("background_subtraction", self.setting_widgets["background_subtraction"])
        background_form.addRow("ballradius", self.setting_widgets["ballradius"])
        background_form.addRow("gradient", self.setting_widgets["gradient"])
        background_form.addRow("gradtype", self.setting_widgets["gradtype"])
        parent_layout.addWidget(background_group)

        detector_group = QGroupBox("Detector")
        detector_form = QFormLayout(detector_group)
        self.setting_widgets["material"] = self._combo(["NaI", "CsI"])
        self.setting_widgets["average_thickness"] = QLineEdit()
        self.setting_widgets["density_NaI"] = QLineEdit()
        self.setting_widgets["density_CsI"] = QLineEdit()
        detector_form.addRow("material", self.setting_widgets["material"])
        detector_form.addRow("average_thickness", self.setting_widgets["average_thickness"])
        detector_form.addRow("density NaI", self.setting_widgets["density_NaI"])
        detector_form.addRow("density CsI", self.setting_widgets["density_CsI"])
        parent_layout.addWidget(detector_group)

        id_group = QGroupBox("Identification")
        id_form = QFormLayout(id_group)
        self.setting_widgets["time"] = QLineEdit()
        self.setting_widgets["energy_window_size"] = QLineEdit()
        self.setting_widgets["category"] = QLineEdit()
        self.setting_widgets["common_isotopes"] = QLineEdit()
        id_form.addRow("time", self.setting_widgets["time"])
        id_form.addRow("energy_window_size", self.setting_widgets["energy_window_size"])
        id_form.addRow("category", self.setting_widgets["category"])
        id_form.addRow("common_isotopes (comma-separated)", self.setting_widgets["common_isotopes"])
        parent_layout.addWidget(id_group)

        run_group = QGroupBox("Run Options")
        run_form = QFormLayout(run_group)
        for key, default in {
            "mariscotti": False,
            "identification": True,
            "efficiency_correction": False,
            "useVoigt": False,
            "smooth": True,
        }.items():
            checkbox = QCheckBox()
            checkbox.setChecked(default)
            self.run_option_widgets[key] = checkbox
            run_form.addRow(key, checkbox)
        parent_layout.addWidget(run_group)

    @staticmethod
    def _combo(options):
        combo = QComboBox()
        combo.addItems(options)
        return combo

    def _load_settings_into_widgets(self, settings):
        for key, widget in self.setting_widgets.items():
            if isinstance(widget, QComboBox):
                idx = widget.findText(str(settings.get(key, DEFAULT_SETTINGS.get(key, ""))))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif key == "density_NaI":
                widget.setText(str(settings.get("density", {}).get("NaI", DEFAULT_SETTINGS["density"]["NaI"])))
            elif key == "density_CsI":
                widget.setText(str(settings.get("density", {}).get("CsI", DEFAULT_SETTINGS["density"]["CsI"])))
            elif key == "common_isotopes":
                widget.setText(",".join(settings.get("common_isotopes", DEFAULT_SETTINGS["common_isotopes"])))
            else:
                widget.setText(str(settings.get(key, DEFAULT_SETTINGS.get(key, ""))))

    def _collect_settings_from_widgets(self):
        updated = dict(self.settings)

        for key, widget in self.setting_widgets.items():
            if key in {"density_NaI", "density_CsI"}:
                continue
            if isinstance(widget, QComboBox):
                updated[key] = widget.currentText()
            elif key == "common_isotopes":
                updated[key] = [item.strip() for item in widget.text().split(",") if item.strip()]
            else:
                updated[key] = widget.text().strip()

        density_nai = float(self.setting_widgets["density_NaI"].text().strip())
        density_csi = float(self.setting_widgets["density_CsI"].text().strip())
        updated["density"] = {"NaI": density_nai, "CsI": density_csi}

        for key in [
            "M",
            "c",
            "a",
            "b",
            "time",
            "confidence",
            "intensity_threshold",
            "base_width",
            "gradient",
            "average_thickness",
            "energy_window_size",
        ]:
            updated[key] = float(updated[key])

        for key in ["min_FWHM", "max_FWHM", "FWHM_step", "LLD", "HLD", "num_iters", "ballradius", "smooth_window", "smooth_order"]:
            updated[key] = int(float(updated[key]))

        return updated

    def _apply_settings(self):
        try:
            updated = self._collect_settings_from_widgets()
            save_settings(updated)
            self.settings = load_settings()
            peak_main.update_settings(self.settings)
            self._log("Settings applied.")
            return True
        except Exception as exc:
            self._show_error(f"Invalid settings: {exc}")
            return False

    def _reset_defaults(self):
        self.settings = reset_to_defaults()
        self._load_settings_into_widgets(self.settings)
        peak_main.update_settings(self.settings)
        self._log("Settings reset to defaults.")

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Spectrum File",
            "",
            "Spectrum Files (*.txt *.csv *.npy);;All Files (*)",
        )
        if not file_path:
            return

        self.file_input.setText(file_path)
        self.selected_file = file_path

        try:
            self.raw_spectrum = peak_main.load_spectrum_file(file_path)
            self._plot_spectrum(raw=self.raw_spectrum)
            self._log(
                f"Loaded spectrum: {Path(file_path).name} ({len(self.raw_spectrum)} channels, "
                f"min={np.min(self.raw_spectrum):.4g}, max={np.max(self.raw_spectrum):.4g})"
            )
        except Exception as exc:
            self.raw_spectrum = None
            self._show_error(f"Failed to load spectrum: {exc}")

    def _run_peakid(self):
        if self.raw_spectrum is None:
            self._show_error("Select a spectrum file first.")
            return

        if not self._apply_settings():
            return

        run_opts = {key: widget.isChecked() for key, widget in self.run_option_widgets.items()}

        try:
            df, corrected, fits, background = peak_main.main(
                np.array(self.raw_spectrum, dtype=int),
                mariscotti=run_opts["mariscotti"],
                identification=run_opts["identification"],
                efficiency_correction=run_opts["efficiency_correction"],
                useVoigt=run_opts["useVoigt"],
                smooth=run_opts["smooth"],
            )

            self._populate_table(df)
            self._plot_spectrum(raw=self.raw_spectrum, processed=corrected, fits=fits, background=background)

            if df.empty:
                self._log("Run completed: no peaks found.")
            else:
                self._log(f"Run completed: {len(df)} peaks reported.")
        except Exception:
            self._show_error(traceback.format_exc())

    def _plot_spectrum(self, raw, processed=None, fits=None, background=None):
        self.ax.clear()

        def _prep_series(values, use_log):
            values = np.asarray(values, dtype=float)
            if use_log:
                return np.where(values > 0, values, np.nan)
            return values

        raw = np.asarray(raw, dtype=float)
        finite_raw = raw[np.isfinite(raw)]
        max_raw = float(np.max(finite_raw)) if finite_raw.size else 0.0
        use_log = max_raw > 1.0

        x_raw = peak_main.channel_to_energy(np.arange(len(raw)))
        self.ax.plot(x_raw, _prep_series(raw, use_log), label="Raw Spectrum", alpha=0.7)

        if processed is not None:
            processed = np.asarray(processed, dtype=float)
            x_proc = peak_main.channel_to_energy(np.arange(len(processed)))
            self.ax.plot(x_proc, _prep_series(processed, use_log), label="Processed Spectrum", alpha=0.7)

        if fits is not None and len(fits) > 0:
            fits = np.asarray(fits, dtype=float)
            fits_plot = _prep_series(fits, use_log)
            x_fit = peak_main.channel_to_energy(np.arange(len(fits)))
            self.ax.plot(x_fit, fits_plot, label="Fits", alpha=0.8)

        if background is not None and np.any(background > 0):
            bg = np.asarray(background, dtype=float)
            x_bg = peak_main.channel_to_energy(np.arange(len(bg)))
            self.ax.plot(x_bg, _prep_series(bg, use_log), label="Background", alpha=0.6)

        self.ax.set_yscale("log" if use_log else "linear")
        self.ax.set_xlabel("Energy (keV)")
        self.ax.set_ylabel("Counts")
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _populate_table(self, dataframe):
        if dataframe is None or dataframe.empty:
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        self.results_table.clear()
        self.results_table.setRowCount(len(dataframe))
        self.results_table.setColumnCount(len(dataframe.columns))
        self.results_table.setHorizontalHeaderLabels([str(col) for col in dataframe.columns])

        for row in range(len(dataframe)):
            for col, column_name in enumerate(dataframe.columns):
                value = dataframe.iloc[row][column_name]
                self.results_table.setItem(row, col, QTableWidgetItem(str(value)))

        self.results_table.resizeColumnsToContents()

    def _log(self, message):
        self.status_text.append(message)

    def _show_error(self, message):
        self._log(message)
        QMessageBox.critical(self, "PeakID Error", message)


def run_GUI():
    app = QApplication(sys.argv)
    window = PeakIDApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    run_GUI()
