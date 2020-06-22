# Stephen Duncanson
# stephen.duncanson@uconn.edu
# Function to process raw data obtained from beamtime at European XFEL
# Berrah Lab, proposal 2318

import numpy as np
import random
import scipy as sci
from scipy import signal
import matplotlib.pyplot as plt
import h5py as h5
from karabo_data import open_run, RunDirectory, H5File, by_index, by_id
import math
import xfel_functions as xf
import time
from scipy.optimize import curve_fit
import os
from mpl_toolkits import mplot3d
from matplotlib import rc

rc('font',**{'family':'serif'})


def open_processed_run(run_number):
    '''
    Inputs:
        run_number: the number of the reduced run to open
    Outputs:
        f: h5 file object
    Todo:
        better dir support
        open multiple files?
    Note:
        analouge of open_run but for processed data
        also changed, see the _proc2.h5, this one is for the files created by scripts in proc2
        depending on the run_number parameter which gets passed, it will go to the expected folder in
        /Stephen/(number)/open the .h5 file
        this uses the relative path, so expects to be run within the same directory as the folders 
    '''
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    print(target_file)
    try:
        f = h5.File(target_file, 'r')
    except:
        print("No processed run file found! for run: "+str(run_number))
        return
    return f


def get_tof(charge):
    '''
    Notes:
        Result of linear calibration
        Transform from tof -> mass/charge
        hardcoded for Ar, m=40u
    Inputs:
        charge: The cation we wish to find the associated tof for.
    Outputs:
        tof:    the tof where we should expect to see Ar+charge
                time unit: ns/2
    Todo:
        Add mass parameter to investigate other species
        Fix Time scale? ns/2 -> ns?
        Possibly use a second or third order function to calibrate?
    '''
    mass = 40
    slope = 6685.782541703501
    intercept = 269.49516087660595
    return slope*np.sqrt(mass/charge)+intercept

# build the pulse dict and plot the ions!!

def round_to_closest_bin(n): 
    bin_min = 3600
    bin_max = 6000
    bin_size = 400
    if n > bin_max:
        return bin_max
    elif n < bin_min:
        return bin_min
    # Smaller multiple 
    a = (n // bin_size) * bin_size
    # Larger multiple 
    b = a + bin_size
    # Return of closest of two 
    return int(b if n - a > b - n else a)

def get_pulse_time():
    '''
    Notes:
        Helper function to reduce computation
        Sourced from Edwin's code
        I don't know where he got these hard-coded values
    Inputs:
        None.
    Outputs:
        time of 1 pulse, time unit: ns/2
    '''
    t_min_period = 1760         # t_min_period is when pulses are in consecutive bunches. unit: ns/2
    every_x_pulse_in_train = 24 # no units, an amount
    return t_min_period*every_x_pulse_in_train
# pulse dictionary will hold all the bins
# key = binned intensity
# value = nested core python list of all the peaks
# len of this nested list = # of total runs in this bin, sqrt(n) = uncertainty
# 
pulse_dict = {}
bin_size = 1
pulse_time = get_pulse_time()
epsilon = 3

all_ints = []


runs_at_1555ev = [45, 46, 52, 53, 55]

runs_at_1550ev = [73,71,53,52,57]
runs_at_2100ev = [29,34,35,36,37,38,39]
bin_size = 400
eps = 12 # 6ns each direcion
#runs_at_2100ev = [29,34,35,36,37,38,39,44,45,46]

#intensity_range = xf.get_intensity_range(runs_at_1550ev)
#print(intensity_range)
intensity_range = [3600, 6000]

# 2: We need to create an array of bins using the intensity_range endpoints
# as well as the bin_size that we define up at the top
#bins = xf.get_bins(intensity_range, bin_size)


bins = [4000, 6000]

runs_at_1450ev = [70, 89, 90, 91, 92]
runs_at_1490ev = [69]
runs_at_1520ev = [68]
runs_at_1530ev = [67]
runs_at_1540ev = [59]
runs_at_1544ev = [66]
runs_at_1550ev = [71,73,53,52,57]
runs_at_1555ev = [45, 46, 52, 53, 55, 65, 88]
runs_at_1559ev = [58]
runs_at_1564ev = [61]
runs_at_1569ev = [64, 93]
runs_at_1576ev = [62, 87]
runs_at_1583ev = [63]

colors = ['#073B4C', '#06D6A0','#EF476F','#118AB2','#FFD166']

for run_number in runs_at_1550ev:
    # for each run, open the run
    f = open_processed_run(run_number)
    
    # load all datasets into numpy arrays
    intensity_per_pulse = np.array(f['intensity_per_pulse'])
    all_ints.append(intensity_per_pulse)
    f.close()

for run, run_number, run_color in zip(all_ints, runs_at_1550ev, colors):
    plt.hist(run, bins=100, alpha=.6, color=run_color, label="Run "+str(run_number))

plt.title("Runs at 1550eV")
plt.xlabel("Pulse Intensity (Arb. Units)")
plt.ylabel("Number of Shots")
plt.legend()
#for x in bins:
#    plt.axvline(x, color='red', alpha=.6)
plt.show()