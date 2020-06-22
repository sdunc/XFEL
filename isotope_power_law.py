# stephen duncanson

import numpy as np
import xfel_functions as xf
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
import matplotlib as mpl
from pylab import cm
rc('font',**{'family':'serif'})

def fit_function(x, a, b):
	return a*x**b 

def get_ratio_and_error(ar17_count, ar16_count):
	ratio = ar17_count/ar16_count

	ratio_error = 1
	return ratio, ratio_error


def get_count_error(count, intensity_bin):
	# dont include the # of pulses because the count is so much higher 
	#number of pulses:
	number_of_pulses_in_bin = xf.get_pulses_in_bin(peaks_at_1550, intensity_bin)
	number_of_total_counts = count*number_of_pulses_in_bin
	error = (1/np.sqrt(number_of_total_counts))*count
	return error


runs_at_1550ev = [73,71,55,53,52,57]
runs_at_2100ev = [29,34,35,36,37,38,39]
bin_size = 400
eps = 12 # 6ns each direcion
offset = .001
#intensity_range = xf.get_intensity_range(runs_at_1550ev)
#I'll just define it
intensity_range = [3600, 5600]

# 2: We need to create an array of bins using the intensity_range endpoints
# as well as the bin_size that we define up at the top
bins = xf.get_bins(intensity_range, bin_size)
print(bins)

# 3: We get a dictionary of all the peaks for our intensity bins
peaks_at_1550 = xf.get_peaks_at_int(runs_at_1550ev, bins)

# 4: We get another dictionary of the count mode at each intensity bin
count_per_pulse_at_1550 = xf.get_count_per_pulse_at_intensity(peaks_at_1550)

# we use regular lists (not np arrays) bc it is faster to append
x_vals = [] # pulse intensity for the point
y_vals = [] # normalized ratio of ar17/ar16
errors = [] #uncertainty of ratio 
# for the power law stuff

ar_40_counts = []
ar_36_counts = []
ar_38_counts = []


ar40_count_error = []
ar36_count_error = []
ar38_count_error = []



for intensity_bin in count_per_pulse_at_1550:
	# sum over a slice
	ar15_40_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(15, mass=40)-eps):int(xf.get_tof(15,mass=40)+eps)])
	ar15_36_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(15, mass=36)-eps):int(xf.get_tof(15,mass=36)+eps)])
	ar15_38_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(15, mass=38)-eps):int(xf.get_tof(15,mass=38)+eps)])
	ar_40_counts.append(ar15_40_count)
	ar_38_counts.append(ar15_38_count)
	ar_36_counts.append(ar15_36_count)
	#ar11_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(11)-eps):int(xf.get_tof(11)+eps)])
	#ar10_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(10)-eps):int(xf.get_tof(10)+eps)])
	# here is where I will need to look at possion errors
	#ratio, error = get_ratio_and_error(ar17_count, ar16_count)
	#print(ratio)
	#print(error)
	#ratio2 = ar16_count/ar15_count
	x_vals.append(intensity_bin)
	#y_vals.append(ratio)


	ar40_count_error.append(get_count_error(ar15_40_count, intensity_bin))
	ar36_count_error.append(get_count_error(ar15_36_count, intensity_bin))
	ar38_count_error.append(get_count_error(ar15_38_count, intensity_bin))


	#errors.append(error)
	#y2_vals.append(ratio)


# make those lists into numpy arrays, speeds up fit func
x = np.asarray(x_vals, dtype=float)

# lets make the figure
fig = plt.figure()
ax = plt.gca()
plt.title("Isotope Count Dependence on Pulse Intensity")

# make this a semilog scale
plt.yscale('log')
plt.xscale('log')

# define the domain of fit function
domain = np.linspace(start=(bins[0]-(bins[1]-bins[0])),stop=(bins[-1]+(bins[1]-bins[0])), num=500)

# do a fit for 17+
popt, pcov = curve_fit(fit_function, x, ar_40_counts, maxfev=10000, sigma=ar40_count_error)#, absolute_sigma=True)
stdevs = np.sqrt(np.diag(pcov))
plt.plot(domain, fit_function(domain, *popt), 'b-')#, label=r"$40Ar^{15+}$ Fit", alpha=.5)
plt.text(3433.13, 0.0157335,"b="+format(str(popt[1]), '.3')+r"$\pm$"+format(str(stdevs[1]),'.3')+" a="+format(str(popt[0]), '.3')+r"$\pm$"+format(str(stdevs[0]),'.3'), fontsize=10)
plt.errorbar(x,ar_40_counts, yerr=ar40_count_error, ecolor='blue', linestyle='none', fmt='b.', capsize=3, label=r'$40Ar^{15+}$')

popt, pcov = curve_fit(fit_function, x, ar_38_counts, maxfev=10000, sigma=ar38_count_error)#, absolute_sigma=True)
stdevs = np.sqrt(np.diag(pcov))
plt.plot(domain, fit_function(domain, *popt), 'g-')#, label=r"$38Ar^{15+}$ Fit", alpha=.5)
plt.text(3686.17,2.08566e-05, "b="+format(str(popt[1]), '.3')+r"$\pm$"+format(str(stdevs[1]),'.3')+" a="+format(str(popt[0]), '.3')+r"$\pm$"+format(str(stdevs[0]),'.3'), fontsize=10)
plt.errorbar(x,ar_38_counts, yerr=ar38_count_error, ecolor='green', linestyle='none', fmt='g.', capsize=3, label=r'$38Ar^{15+}$')

popt, pcov = curve_fit(fit_function, x, ar_36_counts, maxfev=10000, sigma=ar36_count_error)#, absolute_sigma=True)
stdevs = np.sqrt(np.diag(pcov))
plt.plot(domain, fit_function(domain, *popt), 'r-')#, label=r"$36Ar^{15+}$ Fit", alpha=.5)
plt.text(3330.54,0.000263347, "b="+format(str(popt[1]), '.3')+r"$\pm$"+format(str(stdevs[1]),'.3')+" a="+format(str(popt[0]), '.3')+r"$\pm$"+format(str(stdevs[0]),'.3'), fontsize=10)
plt.errorbar(x,ar_36_counts, yerr=ar36_count_error, ecolor='red', linestyle='none', fmt='r.', capsize=3, label=r'$36Ar^{15+}$')


plt.legend()
plt.xlabel("Pulse Intensity (arb. units)", fontsize='medium')#, fontweight='bold')
plt.ylabel('Count/Pulse', fontsize='medium')#, fontweight='bold')

#plt.tight_layout()
plt.show()
