import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exp_fit(x, a, b, c):
    return a * np.exp(b * x + c)

def log_fit(x, a, b, c):
    return a * np.log(b * x + c)

def fit_curve(x, y, fit, guesses=[1, 1, 1]):
    popt, pcov = curve_fit(fit, x, y, p0=guesses, maxfev=100000)
    return popt


def find_correction(x, material, thickness, density):
    # read in data
    data = pd.read_csv("efficiency_data/" + material + ".csv", header=None, skiprows=1)
    data.columns = ["energy", "efficiency"]

    # calculate mass attenuation coefficient
    mus = data["efficiency"] * density[material]
    atten = thickness * mus
    scale = 1 / (atten / atten.max())

    a, b, c = fit_curve(data["energy"], scale, log_fit)
    y1 = log_fit(x, a, b, c)

    return data, scale, y1, x

def correct_spectrum(spectrum, settings):
    material = settings.get("material", "CsI")
    average_thickness = settings.get("average_thickness", 1.27)
    density = settings.get("density", {"NaI": 3.67, "CsI": 4.51})

    x = np.linspace(1, len(spectrum), len(spectrum) - 1)

    calibration_model = str(settings.get("calibration_model", "linear")).lower()
    if calibration_model == "linear":
        M = settings.get("M", 1.0)
        c = settings.get("c", 0.0)
        x = M * x + c
    elif calibration_model == "parabolic":
        a = settings.get("a", 0.0)
        b = settings.get("b", 1.0)
        c = settings.get("c", 0.0)
        x = a * x**2 + b * x + c
    else:
        print("No channel to energy conversion parameters found, efficiency correction will not be applied.")
        return spectrum

    data, scale, y1, x = find_correction(x, material, average_thickness, density)

    spectrum[1:] = np.around(spectrum[1:] * y1).astype(int)
    return spectrum


if __name__ == "__main__":
    material = "NaI"
    thickness = 0.5     # cm
    density = {"NaI": 3.67, "CsI": 4.51}  # g/cm^3

    data, scale, y1, x = find_correction(np.linspace(1, 4096, 4096 - 1), material, thickness, density)

    plt.plot(x, y1, color="red", label="Log Fit")
    plt.scatter(data["energy"], scale, label="Data")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Scaling Factor")
    #plt.savefig("images/" + material + "_mass_attenuation_coefficient.svg")
    plt.show()
