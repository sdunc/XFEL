# steph

import numpy as np
import xfel_functions as xf
import matplotlib.pyplot as plt


runs_at_1550ev = [73,71,53,52,57]
runs_at_2100ev = [29,34,35,36,37,38,39]
runs_at_1555ev = [45, 46, 52, 53, 55]

peaks_at_1550 = xf.get_all_peaks(runs_at_1550ev)
counts_at_1550 = xf.get_all_count(peaks_at_1550)

#peaks_at_2100 = xf.get_all_peaks(runs_at_2100ev)
#counts_at_2100 = xf.get_all_count(peaks_at_2100)

#peaks_at_1555 = xf.get_all_peaks(runs_at_1555ev)
#counts_at_1555 = xf.get_all_count(peaks_at_1555)

plt.ylabel("Count")
plt.xlabel("Tof ns/2")

for charge in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
    plt.axvline(x=xf.get_tof(charge, mass=40) ,linewidth=10, color='grey', alpha=.3)
    plt.axvline(x=xf.get_tof(charge, mass=36) ,linewidth=10, color='green', alpha=.3)
    plt.axvline(x=xf.get_tof(charge, mass=38) ,linewidth=12, color='red', alpha=.3)


    #plt.text(x=xf.get_tof(charge, mass=40), y=50, s="36Ar+"+str(charge),fontsize=8)

x_s = [0]*len(counts_at_1550)

plt.plot(counts_at_1550,label="1550ev,runs: 73,71,53,52,57")
#plt.plot(counts_at_1555,label='1555ev,runs: 45, 46, 52, 53, 55]')
#plt.plot(counts_at_2100,label='2100ev,runs: 29,34,35,36,37,38,39')
plt.legend()
plt.show()