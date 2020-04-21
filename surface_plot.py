# stephen duncanson
# stephen.duncanson@uconn.edu
# surface plot
# the idea here is to be slow and simple and work

import numpy as np
import xfel_functions as xf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

runs_to_analyze = [55,57,58,59,61,62,64,66,67,71,73,93]

# first step is to get our range of pulse intensities for all the runs
# this range will become the x-axis. 
#I can add padding of 0's to make a square shape if needed.
# the y axis will be the pulse time in half nano seconds
# should be easy to get
# the z values will be the count / num pulses for the associated pulse intensity

max_intensity = xf.get_max_intensity(runs_to_analyze)
empty_peaks_at_int = xf.build_peaks_at_int_dict(max_intensity)
peaks_at_int = xf.get_peaks_at_int(runs_to_analyze, empty_peaks_at_int)
count_per_pulse = xf.get_count_per_pulse_at_intensity(peaks_at_int)

# create 1D array of all counts
flat_counts = []
for i in count_per_pulse:
	flat_counts.extend(count_per_pulse[i])
flat_counts = np.array(flat_counts)

X = np.arange(0, max_intensity+1, 1)
Y = np.arange(0, 4224, 1)
X, Y = np.meshgrid(Y, X)
# reshape into 2D array
# tof axis=1 shape 42240
# intensity axis=0 shape max_intensity+1 because 0 indexed
#Z = np.reshape(flat_counts, (max_intensity+1,42240))

a = np.reshape(flat_counts, (max_intensity+1,42240))
Z = xf.mush_tof(a)

fig = plt.figure()
ax = fig.gca(projection='3d')
#vmax=0.0016,linewidth=0,
surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=1,
                        antialiased=False)
ax.set_zlim3d(0, 0.25)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



