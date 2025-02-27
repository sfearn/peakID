# Peak Finding and Fitting

Contains the code for peak finding and fitting for gamma spectra. Running *main.py* provides a GUI with configurable parameters for peak finding and fitting. The expected file is a .txt or .csv file containing counts in each channel (e.g. for a 4096 channel spectrometer, the file would contain 4096 rows each containing a single integer).

Inside the *main.py* file, the *main* function is called, which creates the GUI and runs the main loop. This also has configurable settings:

```    
   df, corrected_array, fits = main(f,
                                     useVoigt=False,
                                     mariscotti=False,
                                     removeBG=True,
                                     identification=False,
                                     efficiency_correction=False,
                                     smooth=True
                                     )
```

```df``` is a pandas dataframe containing the peak information, ```corrected_array``` is the array with the background removed, and ```fits``` is a list of the y_fit values.

**useVoigt**: If True, the fitting function will use a Voigt profile instead of a Gaussian profile.

**mariscotti**: If True, the fitting function will use the Mariscotti function as described in (M.A. Mariscotti, Nucl. Instrum. Method 50 (1967) 309.), instead of the scipy.signal.find_peaks function.

**identification**: If True, the peaks will be identified using the *identify_peaks* function.

**efficiency_correction**: If True, the efficiency correction will be applied to the peaks.
