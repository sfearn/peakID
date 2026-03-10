import json
from pathlib import Path


SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"


DEFAULT_SETTINGS = {
    "calibration_model": "linear",
    "M": 1.0,
    "c": 0.0,
    "a": 0.0,
    "b": 1.0,
    "min_FWHM": 20,
    "max_FWHM": 150,
    "FWHM_step": 5,
    "time": 0.0,
    "confidence": 1.0,
    "intensity_threshold": 1e-9,
    "LLD": 80,
    "HLD": 4080,
    "num_iters": 2,
    "base_width": 10.0,
    "smooth_window": 30,
    "smooth_order": 2,
    "background_subtraction": "none",
    "ballradius": 50,
    "gradient": 10.0,
    "gradtype": "sqrt",
    "material": "CsI",
    "average_thickness": 1.27,
    "density": {"NaI": 3.67, "CsI": 4.51},
    "energy_window_size": 15.0,
    "category": "All",
    "common_isotopes": [
        "Cs-137",
        "Co-60",
        "U233",
        "Cf252",
        "Cs-134",
        "Bi-207",
        "Bi-214",
        "Pb-207",
        "Pb-210",
        "Pb-214",
        "Sr-85",
        "Ag-110m",
    ],
}


NUMERIC_INT_KEYS = {"min_FWHM", "max_FWHM", "FWHM_step", "LLD", "HLD", "num_iters", "ballradius", "smooth_window", "smooth_order"}
NUMERIC_FLOAT_KEYS = {
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
}


VALID_CALIBRATION_MODELS = {"linear", "parabolic", "none"}
VALID_BACKGROUND_SUBTRACTION = {"none", "rolling_ball", "linear"}
VALID_GRADTYPE = {"sqrt", "linear"}


def _coerce_settings(settings):
    coerced = dict(DEFAULT_SETTINGS)
    coerced.update(settings or {})

    for key in NUMERIC_INT_KEYS:
        coerced[key] = int(coerced[key])

    for key in NUMERIC_FLOAT_KEYS:
        coerced[key] = float(coerced[key])

    coerced["calibration_model"] = str(coerced["calibration_model"]).lower()
    if coerced["calibration_model"] not in VALID_CALIBRATION_MODELS:
        coerced["calibration_model"] = DEFAULT_SETTINGS["calibration_model"]

    coerced["background_subtraction"] = str(coerced["background_subtraction"]).lower()
    if coerced["background_subtraction"] not in VALID_BACKGROUND_SUBTRACTION:
        coerced["background_subtraction"] = DEFAULT_SETTINGS["background_subtraction"]

    coerced["gradtype"] = str(coerced["gradtype"]).lower()
    if coerced["gradtype"] not in VALID_GRADTYPE:
        coerced["gradtype"] = DEFAULT_SETTINGS["gradtype"]

    if not isinstance(coerced["density"], dict):
        coerced["density"] = dict(DEFAULT_SETTINGS["density"])

    if not isinstance(coerced["common_isotopes"], list):
        coerced["common_isotopes"] = list(DEFAULT_SETTINGS["common_isotopes"])

    return coerced


def load_settings(settings_file=SETTINGS_FILE):
    settings_path = Path(settings_file)
    if not settings_path.exists():
        save_settings(DEFAULT_SETTINGS, settings_path)
        return dict(DEFAULT_SETTINGS)

    with settings_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    settings = _coerce_settings(loaded)
    if settings != loaded:
        save_settings(settings, settings_path)
    return settings


def save_settings(settings, settings_file=SETTINGS_FILE):
    settings_path = Path(settings_file)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _coerce_settings(settings)
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)


def reset_to_defaults(settings_file=SETTINGS_FILE):
    save_settings(DEFAULT_SETTINGS, settings_file)
    return dict(DEFAULT_SETTINGS)
