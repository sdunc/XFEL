# stephen duncanson

import numpy as np
import xfel_functions as xf
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
import matplotlib as mpl
from pylab import cm
rc('font',**{'family':'sans-serif'})

def fit_function(x, a, m):
	# a*x**b
	return a*x**m


runs_at_1550ev = [73,71,55,53,52,57]
runs_at_2100ev = [29,34,35,36,37,38,39]
bin_size = 400
eps = 12 # 6ns each direcion

#intensity_range = xf.get_intensity_range(runs_at_1550ev)
#I'll just define it
intensity_range = [2000, 6000]

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

# for the power law stuff
ar17_counts = []
ar16_counts = []
ar15_counts = []
ar14_counts = []
ar13_counts = []
ar12_counts = []
ar11_counts = []
ar10_counts = []

for intensity_bin in count_per_pulse_at_1550:
	# sum over a slice
	ar17_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(17)-eps):int(xf.get_tof(17)+eps)])
	ar16_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(16)-eps):int(xf.get_tof(16)+eps)])
	ar15_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(15)-eps):int(xf.get_tof(15)+eps)])
	ar14_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(14)-eps):int(xf.get_tof(14)+eps)])
	ar13_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(13)-eps):int(xf.get_tof(13)+eps)])
	ar12_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(12)-eps):int(xf.get_tof(12)+eps)])
	ar11_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(11)-eps):int(xf.get_tof(11)+eps)])
	ar10_count = np.sum(count_per_pulse_at_1550[intensity_bin][int(xf.get_tof(10)-eps):int(xf.get_tof(10)+eps)])
	# here is where I will need to look at possion errors
	#ratio = ar17_count/ar16_count
	x_vals.append(intensity_bin)
	#y_vals.append(ratio)

	# add the counts to the lists
	ar17_counts.append(ar17_count)
	ar16_counts.append(ar16_count)
	ar15_counts.append(ar15_count)
	ar14_counts.append(ar14_count)
	ar13_counts.append(ar13_count)
	ar12_counts.append(ar12_count)
	ar11_counts.append(ar11_count)
	ar10_counts.append(ar10_count)

# make those lists into numpy arrays, speeds up fit func
x = np.asarray(x_vals, dtype=float)
#y = np.asarray(y_vals, dtype=float)

# lets make the figure
fig = plt.figure()
ax = plt.gca()

# make this a semilog scale
plt.yscale('log')
plt.xscale('log')

# define the domain of fit function
#domain = np.linspace(start=(bins[0]-(bins[1]-bins[0])),stop=(bins[-1]+(bins[1]-bins[0])), num=500)

# fits
#popt, pcov = curve_fit(fit_function, x, ar17_counts, p0=[0,2], maxfev=10000, bounds=([-np.inf,1],[0,np.inf]))
#stdevs = np.sqrt(np.diag(pcov))
#print(stdevs)
#print(popt)

plt.scatter(x_vals,ar17_counts, s=20, color='black', label='ar17+')
plt.scatter(x_vals,ar16_counts, s=20, color='blue', label='ar16+')
plt.scatter(x_vals,ar15_counts, s=20, color='red', label='ar15+')
plt.scatter(x_vals,ar14_counts, s=20, color='purple', label='ar14+')
plt.scatter(x_vals,ar13_counts, s=20, color='orange', label='ar13+')
plt.scatter(x_vals,ar12_counts, s=20, color='green', label='ar12+')


#plt.scatter(x_vals,y_vals, s=20, color='black', label='1550ev')
#plt.plot(domain, fit_function(domain, *popt), 'k--')

plt.xlabel("Pulse Intensity (arb. units)", fontsize='medium')#, fontweight='bold')
#plt.ylabel(r'Normalized $Ar^{17+}$/$Ar^{16+}$ ratio', fontsize='medium')#, fontweight='bold')
plt.ylabel('Count/pulse')#, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.show()
