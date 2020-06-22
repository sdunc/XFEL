# stephen duncanson
# count with variable width bins

import numpy as np
import xfel_functions as xf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rc
rc('font',**{'family':'serif'})

def fit_function(x, a, b):
	return a*x**b

runs_at_1550ev = [73,71,55,53,52,57]

bin_size = 400
eps = 12 # 6ns each direcion
dm = 2
counts_per_bin = 15

minimum_intensity = 2000 # this is the intensity that we start our crawl at

# 3: We get a dictionary of all the peaks for our intensity bins
peaks_at_1550 = xf.get_peaks_at_variable_int(runs_at_1550ev, minimum_intensity)

# 4: We get another dictionary of the count mode at each intensity bin
count_per_pulse_at_1550 = xf.get_count_at_variable_int(peaks_at_1550, counts_per_bin)

x_vals = [] # pulse intensity for the point
y_vals = [] # normalized ratio of ar17/ar16
ratio_errors = [] #unc# for the power law stuff

ar_17_counts = []
ar_16_counts = []
ar_17_count_error = []
ar_16_count_error = []

for intensity_bin in count_per_pulse_at_1550:
	number_of_pulses = count_per_pulse_at_1550[intensity_bin][1]
	ar17count = np.sum(count_per_pulse_at_1550[intensity_bin][0][int(xf.get_tof(17, mass=40)-eps):int(xf.get_tof(17,mass=40)+eps)])/number_of_pulses
	ar16count = np.sum(count_per_pulse_at_1550[intensity_bin][0][int(xf.get_tof(16, mass=40)-eps):int(xf.get_tof(16,mass=40)+eps)])/number_of_pulses

	ratio, error = xf.get_variable_ratio_and_error(ar17count, ar16count, number_of_pulses)
	x_vals.append(intensity_bin)
	y_vals.append(ratio)
	ratio_errors.append(error)

x = np.asarray(x_vals, dtype=float)
print(x)

y = np.asarray(y_vals, dtype=float)
yerrs = np.asarray(ratio_errors, dtype=float)

# lets make the figure
fig = plt.figure()
ax = plt.gca()
domain = np.linspace(start=x[0]-bin_size*dm,stop=x[-1]+bin_size*dm, num=500)

popt, pcov = curve_fit(fit_function, x, y, maxfev=20000, sigma=yerrs)
stdevs = np.sqrt(np.diag(pcov))

fit_a = popt[0]
fit_b = popt[1]
a_error = stdevs[0]
b_error = stdevs[1]

label_text = "a={a:.2e}".format(a=fit_a)+u"\u00B1"+"{aerr:.2f}".format(aerr=a_error)+",b={b:.2f}".format(b=fit_b)+u"\u00B1"+"{berr:.2f}".format(berr=b_error)
#/-{aerr:.2f}, b={b:.2f}+/-{berr:.2f}".format(a=fit_a, aerr=fit_b, b=a_error, berr=b_error)

plt.plot(domain, fit_function(domain, *popt), 'k-', alpha=.7, label=label_text)
	
#plt.text(3433.13, 0.0157335,"b="+format(str(popt[1]), '.3')+r"$\pm$"+format(str(stdevs[1]),'.3')+" a="+format(str(popt[0]), '.3')+r"$\pm$"+format(str(stdevs[0]),'.3'), fontsize=10)
plt.errorbar(x, y, yerr=yerrs, ecolor='black', linestyle='none', fmt='k.', capsize=3)#, label=r'$40Ar^{15+}$')




plt.legend()
plt.xlabel("Pulse Intensity (arb. units)", fontsize='medium')#, fontweight='bold')
plt.ylabel('Count/Pulse', fontsize='medium')#, fontweight='bold')
#plt.tight_layout()
plt.title(r"$Ar^{17+}$ / $Ar^{16+}$ Ratio, Fit Function: $y=ax^{b}$")

# added arrows + text


plt.yscale('log')
plt.xscale('log')
plt.show()

