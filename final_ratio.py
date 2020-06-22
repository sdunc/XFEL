# stephen duncanson

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

#intensity_range = xf.get_intensity_range(runs_at_1550ev)
#I'll just define it
intensity_range = [3000, 6000]

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
ratio_errors = [] #uncertainty of ratio 
# for the power law stuff

ar_17_counts = []
ar_16_counts = []
ar_17_count_error = []
ar_16_count_error = []

for intensity_bin in count_per_pulse_at_1550:
	ar17count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(17, mass=40)-eps):int(xf.get_tof(17,mass=40)+eps)])
	ar16count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(16, mass=40)-eps):int(xf.get_tof(16,mass=40)+eps)])
	ratio, error = xf.get_ratio_and_error(ar17count, ar16count, intensity_bin, peaks_at_1550)
	x_vals.append(intensity_bin)
	y_vals.append(ratio)
	ratio_errors.append(error)

# make those lists into numpy arrays, speeds up fit func
x = np.asarray(x_vals, dtype=float)
y = np.asarray(y_vals, dtype=float)
yerrs = np.asarray(ratio_errors, dtype=float)

print(x)
print(y)

# lets make the figure
fig = plt.figure()
ax = plt.gca()

# define the domain of fit function
domain = np.linspace(start=bins[0]-bin_size*dm,stop=bins[-1]+bin_size*dm, num=500)
deltas = np.linspace(start=-0.0001, stop=0.0001,num=22)


popt, pcov = curve_fit(fit_function, x, y, maxfev=20000, sigma=yerrs)#, bounds=([0, 1],[np.inf, np.inf]))#, absolute_sigma=True)
fit_y = fit_function(x, *popt)
chisq = xf.chi_sq(y, fit_y)
fit_a = popt[0]
fit_b = popt[1]


shifted_a = deltas+fit_a

# I can move this into xfel_functions later

shifted_y_fits = []
for a in shifted_a:
	hypo_fit = fit_function(x, a, fit_b)
	shifted_y_fits.append(hypo_fit) # the possible y fit values for this adjustment

chi_sqs = []
for fits in shifted_y_fits:
	csq = xf.chi_sq(y, fits)
	chi_sqs.append(csq)


delta_chi_sq = np.array(chi_sqs) - chisq
shifted_a_squared = np.array(shifted_a)**2
num = np.sum([x**2 for x in shifted_a_squared])
denom = np.sum(shifted_a_squared*delta_chi_sq)

uncert = np.sqrt(num/denom)

print(uncert)
a = 1/(uncert**2)

deltas_sq = np.array(deltas)**2
a_times_deltas_sq = deltas_sq*a

plt.plot(shifted_a, a_times_deltas_sq, 'b.')
plt.show()






shifted_b = deltas+fit_b

stdevs = np.sqrt(np.diag(pcov))

plt.plot(domain, fit_function(domain, *popt), 'k-', alpha=.7, label="a="+format(str(popt[0]), '.3')+r"$\pm$"+format(str(stdevs[0]),'.2')+",b="+format(str(popt[1]), '.3')+r"$\pm$"+format(str(stdevs[1]),'.3'))

#plt.text(3433.13, 0.0157335,"b="+format(str(popt[1]), '.3')+r"$\pm$"+format(str(stdevs[1]),'.3')+" a="+format(str(popt[0]), '.3')+r"$\pm$"+format(str(stdevs[0]),'.3'), fontsize=10)
plt.errorbar(x, y, yerr=yerrs, ecolor='black', linestyle='none', fmt='k.', capsize=3)#, label=r'$40Ar^{15+}$')
print(popt)
print(pcov)
print(stdevs)



plt.legend()
plt.xlabel("Pulse Intensity (arb. units)", fontsize='medium')#, fontweight='bold')
plt.ylabel('Count/Pulse', fontsize='medium')#, fontweight='bold')
#plt.tight_layout()
plt.title(r"$Ar^{17+}$ / $Ar^{16+}$ Ratio, Fit Function: $y=ax^{b}$")

# added arrows + text


plt.yscale('log')
plt.xscale('log')
plt.show()
