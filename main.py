import pandas as pd
import sys
import time
import numpy as np
import scipy
import glob
from scipy.optimize import curve_fit, minimize
import scipy.special as sp
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import file_edit_GUI as GUI
import efficiency_corrections as ec
plt.rcParams.update({'font.size': 18})


def gaussian(x, mu, sig, a):
    """Function which defines a gaussian curve."""
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def voigt(x, mu, sig, gamma, a):
    """Function which defines a voigt curve."""
    return a * sp.voigt_profile(x - mu, sig, gamma)


def linear_channel_to_energy(channel):
    """Completes y = M*x + c conversions for channel numbers."""
    from config import c, M

    return (M * channel) + c


def parabolic_channel_to_energy(channel):
    """Completes y = ax^2 + bx + c conversions for channel numbers."""
    from config import a, b, c

    return (a * channel ** 2) + (b * channel) + c


def channel_to_energy(channels):
        for i in range(2):
            try:
                if i == 0:
                    energies = linear_channel_to_energy(channels)
                    converted = True
                    break
                elif i == 1:
                    energies = parabolic_channel_to_energy(channels)
                    converted = True
                    break
            except ImportError:
                converted = False
                pass

        if converted == False:
            print("Error: No conversion found in config.py file, using channel numbers as energies.")
            return channels
        else:
            return energies


def findintensity(areas):
    """Finds peak intensity relative to other peaks in spectra."""
    try:
        return 100 * (areas / (max(areas)))
    except ValueError:
        return []


def find_nearest(array, value):
    """Finds closest node to specified value in an array and returns the index."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearestX(array, value, x):
    """Finds closest X nodes to specified value in an array and returns the indices."""
    idxs = []
    for i in range(x):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        idxs.append(idx)
        array = np.delete(array, idx)
    return idxs


def most_common(lst):
    """Find most common occurrence in a list."""
    return max(set(lst), key=lst.count)


def coefficient_of_determination(y, y_fit, n):
    """Calculate the adjusted coefficient of determination (adjusted r^2 value) for a given data set and fit."""
    # calculate residual sum of squares
    ss_res = np.sum((y - y_fit) ** 2)

    # calculate total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1E-9

    # calculate r-squared
    r2 = 1 - (ss_res / ss_tot)

    # number of variables in the data
    k = 2

    # return adjusted r^2
    return 1 - (((1 - r2) * (n - 1)) / (n - k - 1))


def second_diff(N, m, z):
    """Finds second difference of spectrum, discrete analogue to second derivative."""

    # empty list with same size as data
    S = np.zeros(len(N))

    # loop over data points and for each take the second difference
    for i in range(len(N)):
        try:
            S[i] = N[i + 1] - 2 * N[i] + N[i - 1]
        # if at the ends of the array need to do different calculation to avoid error
        except IndexError:
            if i == 0:
                S[i] = N[i + 1] - 2 * N[i]
            else:
                S[i] = N[i - 1] - 2 * N[i]

    # smooth the second difference using Mariscotti's values
    for i in range(z):
        S = average(S, m)

    return S


def average(A, m):
    """Smooths input function by averaging neighboring values."""

    # define empty array same length as data
    B = np.zeros(len(A))

    # loop over data points and sum all points m either side of current one
    for i in range(len(A)):
        for j in range(i - m, i + m):
            try:
                B[i] += A[j]
            # if tries to go outside of boundaries of data, do nothing
            except IndexError:
                pass

    return B


def standard_dev_S(N, m, z):
    """Finds standard deviation of second difference."""
    # define empty array same size as data
    F = np.zeros(len(N))

    # loop over data points and take variance of the second difference
    for i in range(len(N)):
        try:
            F[i] = N[i + 1] + 4 * N[i] + N[i - 1]
        # need different calculation at ends of data array to avoid error
        except IndexError:
            if i == 0:
                F[i] = N[i + 1] + 4 * N[i]
            else:
                F[i] = N[i - 1] + 4 * N[i]

    # smoothing
    for i in range(z):
        F = average(F, m)

    # return square root of variance, std deviation
    return np.sqrt(F)


def choose_region_around_index(array, idxs, idx, val):
    """Calculate start and end indices for a slice around a given data point and given the size of the slice.
    Also accounts for index errors possible near the first and last index of the array."""

    try:
        # try to define start and end indices for slice
        start = idxs[idx] - val
        end = idxs[idx] + val
        # if one of the indices is out of bounds need to correct
        if start < 0 or end >= len(array):
            raise ValueError
    except ValueError:
        if idxs[idx] - val < 0:
            start = 0
            end = idxs[idx] + val    # if start is below 0 start slice at 0 to avoid index error
        elif idxs[idx] + val >= len(array):
            start = idxs[idx] - val
            end = len(array) - 1    # similar for end of array, end at the final index instead to avoid error

    return int(round(start)), int(round(end))


def peak_finding(N, min_FWHM, max_FWHM, FWHM_step, confidence=None):
    """Finds peaks based on Mariscotti's method (M.A. Mariscotti, Nucl. Instrum. Method 50 (1967) 309.)."""

    if confidence == None:
        # get confidence factor from config
        from config import confidence, intensity_threshold
    else:
        intensity_threshold = (1e-2 / max(N))

    # define empty array same size as data
    signals = np.zeros(len(N))
    FWHMs = np.zeros(len(N))
    for FWHM in range(min_FWHM, max_FWHM, FWHM_step):
        # empirical values for peak finding algorithm found by Mariscotti
        z = 5
        w = int(0.6 * FWHM)
        if (w % 2) == 0:
            w += 1
        m = int((w - 1) / 2)
        if FWHM < 4:
            continue

        # take second difference and standard deviation of spectrum
        S = second_diff(N, m, z)
        F = standard_dev_S(N, m, z)

        # send spectra from array to a list
        x = N.tolist()

        # smooth
        for i in range(z):
            x = average(x, m)

        starts, ends = [], []

        # if the second difference is negative (indicates gaussian-like feature) and magnitude is greater than
        # the standard deviation multiplied by the confidence factor then can say a peak has been found,
        # if peak is greater than a % of spectrum's intensity, set index of peak centre to 1 in the signals array
        for i in range(len(N)):
            if abs(S[i]) > F[i] * confidence and S[i] < 0:
                if i - FWHM < 0:
                    start = 0
                else:
                    start = i-FWHM
                if i+FWHM>len(x):
                    end = len(x)-1
                else:
                    end = i+FWHM

                if sum(signals[start:end]) > 0:
                    continue

                try:
                    if x[i] == max(x[start:end]) and x[i]/max(x) >= intensity_threshold:
                        signals[np.argmax(x[start:end])+start] = 1
                        FWHMs[np.argmax(x[start:end])+start] = FWHM
                except ValueError as e:
                    print(f"Error: {e, i, FWHM}")

    return signals, FWHMs


def remove_linear_bg(array, locs, FWHMs):
    """Removes straight line background from underneath the peaks."""

    correction = np.zeros(array.shape)
    for i in range(len(locs)):
        start, end = choose_region_around_index(array, locs, i, FWHMs[i])
        if start > 5 and end < len(array) - 6:
            # find mean count near start and end of slice
            y_vals = np.concatenate((array[start - 10:start], array[end:end + 10]))
            x_vals = np.concatenate((np.arange(start - 10, start), np.arange(end, end + 10)))
        elif start <= 5:
            y_vals = np.concatenate((array[0:start], array[end:end + 10]))
            x_vals = np.concatenate((np.arange(0, start), np.arange(end, end + 10)))
        elif end >= len(array) - 6:
            y_vals = np.concatenate((array[start - 10:start], array[end:len(array) - 1]))
            x_vals = np.concatenate((np.arange(start - 10, start), np.arange(end, len(array) - 1)))

        a, b = np.polyfit(x_vals, y_vals, 1)  # fit a straight line to the mean counts

        # define correction as straight line between mean count before and after peak
        x_fit_vals = np.linspace(start, end, end - start)
        correction_vals = a * x_fit_vals + b
        correction[start:end] = correction_vals.astype(int)

    corrected = array - correction  # correct the array

    # make sure none of the array is negative
    if any(a <= 0 for a in corrected):
        for i in range(len(corrected)):
            if corrected[i] <= 0:
                corrected[i] = 0

    return corrected, correction


def radiusheights(radius):
    return [(radius ** 2 - i ** 2) ** 0.5 for i in range(-radius, radius + 1)]


def sqrt_radius(x, a, b):
    return a * x ** 0.5 + b


def linear_radius(x, a, b):
    return a * x + b


def find_height(spectrum, i, radius):
    heights = radiusheights(radius)
    lowerbound = i - radius
    upperbound = i + radius + 1
    if lowerbound < 0:
        heightsubsets = heights[-lowerbound:]
        lowerbound = 0
    elif upperbound > len(spectrum) - 1:
        heightsubsets = heights[: -(upperbound - (len(spectrum)))]
        upperbound = len(spectrum) - 1
    else:
        heightsubsets = heights

    requiredheight = np.inf
    spectrumsubset = spectrum[lowerbound: upperbound + 1]
    for index, height in enumerate(heightsubsets):
        thisheight = spectrumsubset[index] - height
        if thisheight < requiredheight:
            requiredheight = spectrumsubset[index] - height

    return requiredheight


def rolling_ball_bg_subtract(spectrum, startradius, gradient=0, gradtype="sqrt", verbose=False):
    print(f"Rolling ball background subtraction, start radius: {startradius}, gradient: {gradient}, type: {gradtype}")
    background = np.zeros(len(spectrum))
    if gradtype == "sqrt":
        if sqrt_radius(len(spectrum), gradient, startradius) > len(spectrum):
            # raise error
            print("Error: radius too large")
            return np.zeros(len(spectrum))
    elif gradtype == "linear":
        if linear_radius(len(spectrum), gradient, startradius) > len(spectrum):
            # raise error
            print("Error: radius too large")
            return np.zeros(len(spectrum))

    channels = np.arange(len(spectrum))
    energies = channel_to_energy(channels)

    for i in channels:
        if spectrum[i] < 1:
            background[i] = 0

        # change radius as func of channel number or energy
        if gradtype == "sqrt":
            try:
                radius = round(sqrt_radius(energies[i], gradient, startradius))
            except ValueError:
                radius = startradius
        elif gradtype == "linear":
            try:
                radius = round(linear_radius(energies[i], gradient, startradius))
            except ValueError:
                radius = startradius

        if verbose:
            print(f"Radius: {radius}")

        heighthere = find_height(spectrum, i, radius) + radius

        background[i] = heighthere

    print(f"Final radius: {radius}")

    return background


def moving_avg_filter(avg_array, n):
    """Applies a moving average filter to the input array. Window is the size of the filter around central point."""
    return np.convolve(avg_array, np.ones(n) / n, mode='same')


def peak_fitting(array, locs, FWHMs, widths=np.array([]), voigt_func=False):
    """Fits the peaks with a gaussian and returns the fit plus r^2 test statistic."""
    # define empty arrays for r^2 vals and gaussian fits
    new_locs, std_devs, r2s, new_FWHMs, full_widths, peak_areas, errors = [], [], [], [], [], [], []
    fits = np.zeros(array.shape)

    from config import base_width, min_FWHM, max_FWHM

    # cycle through peaks to fit each one
    for i in range(len(locs)):
        # choose region around peak to fit
        if widths.size > 0:
            start, end = choose_region_around_index(array, locs, i, widths[i]/2)    # get start and end indices of slice
        else:
            start, end = choose_region_around_index(array, locs, i, FWHMs[i])

        data = array[start:end]     # take slice of spectrum

        if len(data) < 10 or max(data) < 5:
            continue

        # x values to go into gaussian equation
        x = np.linspace(start, end, end-start)

        # estimated parameters
        mean = locs[i]
        sigma = FWHMs[i] / 2.355
        amp = max(data)
        gamma = 0

        # calculate gaussian fit parameters from given estimated parameters
        try:
            if voigt_func:
                popt, pcov = curve_fit(voigt, x, data, p0=[mean, sigma, gamma, amp])
                gamma = abs(popt[2])
                std_dev = abs(popt[1])
                # FWHM for voigt
                FWHM_g = 2 * np.sqrt(2 * np.log(2)) * abs(std_dev)
                full_width_g = 2 * np.sqrt(2 * np.log(10)) * abs(std_dev)
                FWHM_l = 2 * gamma
                FWHM = (FWHM_l*0.5346) + np.sqrt(0.2166*FWHM_l**2 + FWHM_g**2)
                full_width = (FWHM_l*0.5346) + np.sqrt(0.2166*FWHM_l**2 + full_width_g**2)
                errs = np.sqrt(np.diag(pcov))

            else:
                popt, pcov = curve_fit(gaussian, x, data, p0=[mean, sigma, amp])
                # FWHM for gaussian
                std_dev = abs(popt[1])
                FWHM = std_dev * 2.355
                errs = np.sqrt(np.diag(pcov))

                full_width = std_dev * np.sqrt(2 * np.log(base_width)) * 2
            new_loc = popt[0]

            # create voigt or gaussian fit
            if voigt_func:
                y_fit = voigt(x, *popt)
            else:
                y_fit = gaussian(x, *popt)
            fits[start:end] += y_fit

            r2 = coefficient_of_determination(data, y_fit, len(data))
        except Exception as e:
            print(e)
            FWHM = 0
            r2 = 0

        if widths.size > 0:
            # if the fit is bad, reject the peak
            if i > 0:
                diff = locs[i] - locs[i-1]
            else:
                diff = np.inf
            if not r2 < 0.6 and min_FWHM < FWHM < max_FWHM and diff > FWHM:
                try:
                    new_locs.append(new_loc)
                    r2s.append(r2)
                    new_FWHMs.append(FWHM)
                    full_widths.append(full_width)
                    peak_areas.append(np.sum(array[int(round(new_loc)-round(std_dev)):int(round(new_loc)+round(std_dev))])/0.67)
                    peak_areas.append(np.sqrt(peak_areas[-1]))
                    errors.append(errs)
                except Exception as e:
                    print(e)
                    fits[start:end] = 0
            else:
                print(f"Peak at {locs[i]} rejected: {locs[i]-locs[i-1]} > {FWHM}, r2={r2},FWHM={FWHM}")
                fits[start:end] = 0
        else:
            if r2 > 0.1:
                try:
                    new_locs.append(new_loc)
                    r2s.append(r2)
                    new_FWHMs.append(FWHM)
                    full_widths.append(full_width)
                    peak_areas.append(
                        np.sum(array[int(round(new_loc) - round(std_dev)):int(round(new_loc) + round(std_dev))]) / 0.67)
                    peak_areas.append(np.sqrt(peak_areas[-1]))
                    errors.append(errs)
                except Exception as e:
                    print(e)
                    fits[start:end] = 0
            else:
                print(f"Peak at {locs[i]} rejected: {array[start]} > {array[(end-start)//2]} > {array[end]}, r2={r2}, FWHM={FWHM}")
                fits[start:end] = 0

    return (np.array(new_locs), np.array(r2s), fits, np.array(peak_areas), np.array(new_FWHMs), np.array(full_widths),
            np.array(errors))


def source_lookup(Es, r2s, peak_areas, corrected):
    """Reads lookup table and attempts to match peaks to isotopes.

    Parameters:
    Es (list): list of peak energies
    r2s (list): list of r^2 values for each peak
    peak_areas (list): list of peak areas for each peak
    corrected (bool): whether the spectrum has been efficiency corrected

    Returns:
    list: list of identified peaks
    """
    try:
        from config import time
        if time < 1:
            raise ImportError
    except ImportError:
        print("Valid time value not provided, CPS reported will be total counts")
        time = 1

    from config import category, energy_window_size

    energy_ref = pd.read_csv("IsotopeLibrary.csv")  # read energy reference
    df = pd.DataFrame(energy_ref)   # turn energy ref into pandas data frame
    d = {}

    # convert the peak areas to relative intensities and their errors as well as counts per second for each peak
    intensities = findintensity(peak_areas)
    Is = intensities[::2]
    I_errs = intensities[1::2]
    cps = peak_areas[::2] / time

    # generate empty lists of values
    sources = [0] * len(Es)
    isotopes = [0] * len(Es)
    absoluteIs = [0] * len(Is)
    ACFs = [0] * len(Is)

    # add all isotopes with a peak within 5% of calculated peak energy to a dataframe with unique name
    # for each identified peak this is then added to a dictionary for use when trying to identify which isotope
    # is responsible for each peak
    if category == "All":
        for i in range(len(Es)):
            i_list = []
            for j in range(len(df["Energy (keV)"])):
                if Es[i] - energy_window_size <= df["Energy (keV)"][j] <= Es[i] + energy_window_size:
                    i_list.append(df.loc[j])
            d["df" + str(i)] = pd.DataFrame.from_records(i_list, columns=df.columns)
    else:
        for i in range(len(Es)):
            i_list = []
            for j in range(len(df["Energy (keV)"])):
                if Es[i] - energy_window_size <= df["Energy (keV)"][j] <= Es[i] + energy_window_size \
                        and df["Category"][j] == category:
                    i_list.append(df.loc[j])
            d["df" + str(i)] = pd.DataFrame.from_records(i_list, columns=df.columns)

    # add all dataframes in the dicitonary together (one big list of all possible peaks for the identified peaks)
    dfx = pd.DataFrame(columns=df.columns)
    for i in range(len(Es)):
        if dfx.empty:
            dfx = d["df" + str(i)]
        elif not d["df" + str(i)].empty:
            dfx = pd.concat([dfx, d["df" + str(i)]], ignore_index=True)

    # try to identify the isotopes responsible for each peak
    for i in range(len(Es)):
        sources[i], isotopes[i], absoluteIs[i], ACFs[i] = find_optimum_source(d, i, Es, Is)

    # if multiple peaks and the spectrum is efficiency corrected, check the intensity ratios of the peaks
    if len(Es) > 1 and corrected:
        Es, Is, I_errs, absoluteIs, isotopes, sources, cps, r2s, ACFs, d, hidden_peaks = \
            check_ratios(Es, Is, absoluteIs, I_errs, isotopes, sources, cps, r2s, ACFs, d)

        # if hidden peaks found need to reassign isotopes to the peaks
        if hidden_peaks == True:
            for i in range(len(Es)):
                sources[i], isotopes[i], absoluteIs[i], ACFs[i] = find_optimum_source(d, i, Es, Is)

    # return the data for the output table
    identified_peaks = []
    for i in range(len(Es)):
        try:
            identified_peaks.append([sources[i], round(Es[i], 2),
                                    str(np.round(Is[i], 2)) + u"\u00B1" +
                                    str(np.round(I_errs[i], 2)),
                                    round(r2s[i], 4), round(cps[i], 2), ACFs[i]])
        except:
            print(I_errs[i])

    return identified_peaks


def find_optimum_source(d, i, Es, Is):
    """Finds the most likely source causing a peak based on different criteria."""
    from config import common_isotopes

    # if no sources in dataframe for this peak then skip it
    dfi = d["df" + str(i)]
    if len(dfi["Isotope"]) == 0:
        return 0, 0, 0, 0

    # empty lists to be added to
    sources, isotope_list, absoluteIs = [], [], []

    # first check if multiple peaks could be same source (are there same isotopes in multiple of the distinct dfs)
    if len(Es) > 1:
        source_list = []
        for j in range(len(Es)):
            if i != j:
                dfj = d["df" + str(j)]
                list_i = set(dfi["Isotope"])
                list_j = set(dfj["Isotope"])
                for l in list_i:
                    for k in list_j:
                        if l == k:
                            source_list.append(l)

        # try assigning the most common occurrence in list of isotopes occurring in multiple of the distinct dfs
        try:
            isotope_list.append(most_common(source_list))
            for j in range(len(dfi["Isotope"])):
                if dfi["Isotope"][j] == isotope_list[0]:
                    sources.append(dfi["Isotope"][j] + " (" + dfi["Sub-category"][j] + ")")
                    absoluteIs.append(dfi["Absolute emission probability (%)"][j])
        except ValueError:
            pass

    # then see if any of the common isotopes could be attributed
    if any(a in common_isotopes for a in dfi["Isotope"]):
        temp_Es, temp_Is, temp_isotopes, temp_sources = [], [], [], []
        for j in range(len(dfi["Energy (keV)"])):
            if dfi["Isotope"][j] in common_isotopes:
                temp_Es.append(dfi["Energy (keV)"][j])
                temp_Is.append(dfi["Absolute emission probability (%)"][j])
                temp_isotopes.append(dfi["Isotope"][j])
                temp_sources.append(dfi["Isotope"][j] + " (" + d["df" + str(i)]["Sub-category"][j] + ")")

        # try assign common isotope with peak closest in energy to the identified peak
        if len(temp_Es) > 0:
            index = find_nearest(temp_Es, Es[i])
            sources.append(temp_sources[index])
            isotope_list.append(temp_isotopes[index])
            absoluteIs.append(temp_Is[index])

    # then try to find isotope with peak closest to the found peak's energy
    if len(dfi["Energy (keV)"]) > 3:
        closest_Es = find_nearestX(dfi["Energy (keV)"], Es[i], 3)
        closest_Is = find_nearestX(dfi["Absolute emission probability (%)"], Is[i], 3)
        idx = None
        for a in closest_Es:
            if a in closest_Is:
                idx = a

        if idx == None:
            idx = closest_Es[0]

        sources.append(dfi["Isotope"][idx] + " (" + dfi["Sub-category"][idx] + ")")
        isotope_list.append(dfi["Isotope"][idx])
        absoluteIs.append(dfi["Absolute emission probability (%)"][idx])
    else:
        idx = find_nearest(dfi["Energy (keV)"], Es[i])
        sources.append(dfi["Isotope"][idx] + " (" + dfi["Sub-category"][idx] + ")")
        isotope_list.append(dfi["Isotope"][idx])
        absoluteIs.append(dfi["Absolute emission probability (%)"][idx])

    # how many of the above methods agree on the isotope determines the confidence factor
    arbitrary_confidence_factor = 1
    if len(isotope_list) == 3:
        if len(set(isotope_list)) == 1:
            arbitrary_confidence_factor = 3
        elif len(set(isotope_list)) == 2:
            arbitrary_confidence_factor = 2
    elif len(isotope_list) == 2:
        if len(set(isotope_list)) == 1:
            arbitrary_confidence_factor = 2
    else:
        arbitrary_confidence_factor = 1

    # return the first one in the list as this is probably the best answer
    return sources[0], isotope_list[0], absoluteIs[0], arbitrary_confidence_factor


def check_ratios(Es, Is, absoluteIs, I_errs, isotopes, sources, cps, r2s, ACFs, d):
    """check peak intensity ratios and attempt to find hidden peaks"""
    # assume no hidden peaks initially
    hidden_peaks = False

    # for each peak calculate the ratio of intensities and compare to the expected ratio of intensities
    for i in range(len(Es)):
        for j in range(len(Es)):
            if isotopes[i] == isotopes[j] and i != j and isotopes[i] != 0:
                A = absoluteIs[i] / absoluteIs[j]
                B = Is[i] / Is[j]#

                # if one is found need to add the extra hidden peak to the lists of quantities
                if A - (A * 0.4) <= B <= A + (A * 0.4):
                    pass
                else:
                    hidden_peaks = True
                    if A > 1 and B > 1 or A < 1 and B < 1:
                        A_to_R = A / B
                        if A_to_R > 1:
                            Es = np.append(Es, Es[j])
                            Is = np.append(Is, Is[j] - (Is[j] / A_to_R))
                            Is[j] = Is[j] / A_to_R
                            d["df" + str(len(Es) - 1)] = d["df" + str(j)]
                            isotopes.append(0)
                            sources.append(0)
                            absoluteIs.append(0)
                            r2s = np.append(r2s, r2s[j])
                            cps = np.append(cps, cps[j])
                            I_errs = np.append(I_errs, I_errs[j])
                            ACFs.append(1)
                            break
                        else:
                            Es = np.append(Es, Es[i])
                            Is = np.append(Is, Is[i] - (Is[i] * A_to_R))
                            Is[i] = Is[i] * A_to_R
                            d["df" + str(len(Es) - 1)] = d["df" + str(i)]
                            isotopes.append(0)
                            sources.append(0)
                            absoluteIs.append(0)
                            r2s = np.append(r2s, r2s[i])
                            cps = np.append(cps, cps[i])
                            I_errs = np.append(I_errs, I_errs[i])
                            ACFs.append(1)
                            break
                    else:
                        pass
            break

    return Es, Is, absoluteIs, I_errs, isotopes, sources, cps, r2s, ACFs, d, hidden_peaks


def main(spec_array, mariscotti=False, identification=False, efficiency_correction=False,
         useVoigt=False, smooth=False):
    """Main function for processing spectra. Must be passed a numpy array."""
    # try to retrieve config values and exit if not found
    try:
        from config import (min_FWHM, max_FWHM, FWHM_step, HLD, LLD, intensity_threshold, background_subtraction,
                            num_iters, smooth_window, smooth_order)
    except Exception as e:
        print(f"FWHM estimates, HLD and LLD must be provided in config.py, error: {e}")
        sys.exit()

    # if the discriminators are outside of range of data, can't use them so just use all data
    if HLD > len(spec_array):
        print("HLD not in range of spectrum, will be disregarded")
        HLD = len(spec_array) - 1
    if LLD < 0:
        print("LLD not in range of spectrum, will be disregarded")
        LLD = 0

    if efficiency_correction:
        eff_corrected_array = ec.correct_spectrum(spec_array)
    else:
        eff_corrected_array = spec_array

    if smooth:
        spec = savgol_filter(eff_corrected_array, smooth_window, smooth_order)  # smooth spectrum
    else:
        spec = eff_corrected_array
    corrected_array = np.clip(spec.astype(int), 0, None)  # clip negative values to 0
    bg_corrected_array = np.zeros(corrected_array.shape)

    if background_subtraction == "rolling_ball":
        try:
            from config import ballradius, gradient, gradtype
        except ImportError:
            print("Error: rolling ball params not provided in config.py, defaulting to 500 size ball (no gradient)")
            ballradius = 500
            gradient = 0
            gradtype = "sqrt"

        background = rolling_ball_bg_subtract(corrected_array, ballradius, gradient, gradtype)
        bg_corrected_array += corrected_array - background
        bg_corrected_array[bg_corrected_array < 0] = 0
    else:
        background = np.zeros(len(corrected_array))
        bg_corrected_array += corrected_array

    if mariscotti:
        signals, FWHMs = peak_finding(bg_corrected_array, min_FWHM, max_FWHM, FWHM_step)
        # find the indices of the peaks
        peak_indices = np.where(signals[LLD:HLD] != 0)[0] + LLD
        FWHMs = FWHMs[peak_indices].astype(int)
    else:
        from config import intensity_threshold
        peak_indices = find_peaks(bg_corrected_array[LLD:HLD], prominence=10)[0] + LLD
        FWHMs = [int((max_FWHM-min_FWHM)/2),]*len(peak_indices)

    #check peaks are not too close together
    delete_idxs = []
    for i in range(len(peak_indices)-1):
        if peak_indices[i+1] - peak_indices[i] < FWHMs[i]:
            delete_idxs.append(i)
    peak_indices = np.delete(peak_indices, delete_idxs)
    FWHMs = np.delete(FWHMs, delete_idxs)

    print(f"Found {len(peak_indices)} peaks at indices {peak_indices} with FWHMs {FWHMs}")

    if background_subtraction == "linear":
        bg_corrected_array, correction = remove_linear_bg(bg_corrected_array, peak_indices, FWHMs)

    full_widths = np.array([])
    for i in range(num_iters):
        # fit the peaks with gaussian and return goodness of fits, areas and FWHMs
        peak_indices, goodness_of_fits, fits, peak_areas, FWHMs, full_widths, errs = \
            peak_fitting(bg_corrected_array, peak_indices, FWHMs, widths=full_widths, voigt_func=useVoigt)

    energies = channel_to_energy(peak_indices)
    try:
        if energies[0] != peak_indices[0]:
            print(f"peaks found at energies: {energies} with FWHMs: {FWHMs} and goodness of fits: {goodness_of_fits}")
        else:
            print(f"Peaks found at indices: {peak_indices} with FWHMs: {FWHMs} and goodness of fits: {goodness_of_fits}")
    except Exception as e:
        print(f"Error: {e}")
        if len(energies) == 0 or len(peak_indices) == 0:
            print("No peaks found")
            sys.exit()
    converted_FWHMs = channel_to_energy(FWHMs)
    converted_full_widths = channel_to_energy(full_widths)

    if identification:
        identified_peaks = source_lookup(energies, goodness_of_fits, peak_areas, efficiency_correction)
        df = pd.DataFrame(identified_peaks, columns=["Isotope", "Energy (keV)", "RI (%)",
                                                     r"$\bar{R}^2$ of fit", "CPS", "ACF"])
    else:
        areas = peak_areas[0::2]
        area_errors = peak_areas[1::2] / goodness_of_fits

        data = {'Energy': energies, 'Peak Area': areas, 'Peak Area Errors': area_errors,
                'FWHM': converted_FWHMs, 'Full Width': converted_full_widths, 'Goodness of Fit': goodness_of_fits}
        df = pd.DataFrame(data)

    return df, bg_corrected_array, fits, background


if __name__ == "__main__":
    #### EXAMPLE CODE FOR PROCESSING ####
    start = time.time()

    # define file
    file = GUI.run_GUI()
    if file is None:
        sys.exit()

    # read file as numpy array
    if file[-4:] == ".txt":
        f = np.genfromtxt(file, delimiter="\n", dtype=int)
    elif file[-4:] == ".npy":
        f = np.load(file)
    elif file[-4:] == ".csv":
        f = np.genfromtxt(file, delimiter=",", dtype=int)
    else:
        try:
            f = np.genfromtxt(file, dtype=int)
        except:
            print("File type not recognised")
            sys.exit()

    df, corrected_array, fits, bg = main(f,
                                     useVoigt=False,
                                     mariscotti=False,
                                     identification=True,
                                     efficiency_correction=False,
                                     smooth=True
                                     )

    df.to_csv(f"{file[:-4]}_info.csv", index=False)

    for i in range(len(fits)):
        if fits[i] < 1:
            fits[i] = float('nan')
        if corrected_array[i] < 1:
            corrected_array[i] = 1

    from config import LLD, HLD

    if np.all(bg == 0):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(channel_to_energy(np.arange(len(f))), f, label="Raw Spectrum", alpha=0.5)
        ax.plot(channel_to_energy(np.arange(len(corrected_array))), corrected_array,
                label="Processed Spectrum", alpha=0.5)
        ax.plot(channel_to_energy(np.arange(len(fits))), fits, label="Fits", alpha=0.5)
        ax.legend()
        ax.set_ylim(1, None)
        ax.set_yscale("log")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        plt.show()
    else:
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        ax[0].plot(channel_to_energy(np.arange(len(f))), f, label="Raw Spectrum", alpha=0.5)
        ax[0].plot(channel_to_energy(np.arange(len(corrected_array))), corrected_array,
                   label="Processed Spectrum", alpha=0.5)
        ax[0].plot(channel_to_energy(np.arange(len(fits))), fits, label="Fits", alpha=0.5)
        ax[0].legend()
        ax[0].set_ylim(1, None)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Energy (keV)")
        ax[0].set_ylabel("Counts")
        ax[1].plot(channel_to_energy(np.arange(len(bg))), bg, label="Background", alpha=0.5)
        ax[1].legend()
        #ax[1].set_ylim(1, None)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("Energy (keV)")
        ax[1].set_ylabel("Counts")
        plt.show()
