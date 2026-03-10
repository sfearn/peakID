# Peak Finding and Fitting

PeakID provides peak finding, fitting, and isotope lookup for gamma spectra.

## GUI workflow (PyQt6)

Run `main.py` to open the GUI. The app now:

- Uses grouped settings panels (Calibration, Peak Finding, Fitting/Smoothing, Background, Detector, Identification, Run Options)
- Includes one `Apply Settings` button and one `Reset to Defaults` button
- Loads and displays the selected spectrum immediately
- Runs processing without closing the window
- Shows results in a table inside the GUI
- Keeps the GUI open and retains the selected spectrum even when no peaks are found

## Settings storage

Settings are stored in `settings.json` and managed by `settings_manager.py`.

- `config.py` is no longer required by the processing or GUI pipeline
- If `settings.json` does not exist, defaults are created automatically

## Programmatic usage

The processing entry point remains `main.main(...)`:

```
df, corrected_array, fits, background = main(
   spectrum,
   useVoigt=False,
   mariscotti=False,
   identification=True,
   efficiency_correction=False,
   smooth=True,
)
```

- `df`: peak table (can be empty if no peaks are found)
- `corrected_array`: processed spectrum
- `fits`: fitted profile array
- `background`: estimated background
