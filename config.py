# parameters important to the peak finding and ID can be easily changed in this file

# detector/spectrum specific:
M, c = 1, 0

# range of FWHM to test for peak finding and increment to go up in
min_FWHM, max_FWHM, FWHM_step = 20, 150, 5

# time over which spectrum accumulated
time = 0

# parameters to adjust sensitivity of peak finding, lowering confidence and intensity threshold will find peaks
# more easily but may identify some non-peak features as peaks additionally:
confidence = 1  # confidence factor, f, for second difference method of peak finding
intensity_threshold = 1E-9  # threshold for height of detected peaks in %
LLD = 80  # lower channel number threshold, everything below will be ignored
HLD = 4080  # higher channel number threshold, everything above will be ignored

# fitting parameters
num_iters = 2   # number of iterations for peak fitting - more may make fits worse
base_width = 10  # this is used to calculate width at 1/x of peak height for peak fitting after first iteration

#smoothing parameters
smooth_window = 30  # window size for smoothing spectrum
smooth_order = 2  # order of polynomial for smoothing spectrum

background_subtraction = "none"  # options are "rolling_ball", "linear", "none"
ballradius = 50 # radius of rolling ball for background subtraction
gradient = 10  # gradient of ball radius with channel number
gradtype = "sqrt" # type of relationship of ball radius with energy/channel number, options are "linear" or "sqrt"

# detector parameters
material = "CsI"  # options are NaI, CsI - add data files for other materials if needed
average_thickness = 1.27  # average path of gamma through detector in cm
density = {"NaI": 3.67, "CsI": 4.51}  # g/cm^3

# isotope ID parameters:
energy_window_size = 15  # number of keV to include isotopes with energies +/- of detected peak energy
category = "All"  # category of isotopes to include in ID, options are Nuclear Power Plant, Homeland Security, Early Nuclear Accident, Late-Time Nuclear Accident, Spent Fuel, CTBT OSI, Laboratory Miscellaneous, All

# isotopes commonly found in spectra, given priority when performing ID
common_isotopes = ["Cs-137", "Co-60", "U233", "Cf252", "Cs-134", "Bi-207", "Bi-214", "Pb-207", "Pb-210", "Pb-214",
                   "Sr-85", "Ag-110m"]