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
#intensity_range = [3600, 6000]

# 2: We need to create an array of bins using the intensity_range endpoints
# as well as the bin_size that we define up at the top
#bins = xf.get_bins(intensity_range, bin_size)

bins = [(3400, 3800),(3800, 4200),(4200, 4600),(4600, 5000),(5000, 5400),(5400, 5800), (5800, 6200)]
colors = ['#3498DB', '#17A589','#F1C40F','#F39C12','#D35400','#C0392B','#884EA0']
labels =[r'335199 Pulses, 11 $Ar^{17+}$ 1006 $Ar^{16+}$',\
        r'678492 Pulses, 34 $Ar^{17+}$ 3161 $Ar^{16+}$',\
        r'378670 Pulses, 34 $Ar^{17+}$ 2924 $Ar^{16+}$',\
        r'135783 Pulses, 35 $Ar^{17+}$ 2026 $Ar^{16+}$',\
        r'29765 Pulses, 13 $Ar^{17+}$ 805 $Ar^{16+}$',\
        r'4460 Pulses, 7 $Ar^{17+}$ 204 $Ar^{16+}$',\
        r'341 Pulses, 1 $Ar^{17+}$ 25 $Ar^{16+}$',\

        ]
#bins = [4000, 6000]

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

for run_number in runs_at_1540ev:
    # for each run, open the run
    f = open_processed_run(run_number)
    
    # load all datasets into numpy arrays
    intensity_per_pulse = np.array(f['intensity_per_pulse'])
    all_ints.extend(intensity_per_pulse)
    f.close()


plt.hist(all_ints, bins=100, alpha=.7, color="#566573")
plt.title("Cumulative Pulses at 1550eV")
plt.xlabel("Pulse Intensity (Arb. Units)")
plt.ylabel("Number of Shots")

binnin = [2814,3714,3843.5,3948.5,4062.,4178.5,4306.5,4430.5,4548.,4657.5,4783.5,4908.,5055.5,5377.,5828.93811881]
plt.axvline(2000, alpha=.5, color='green', label='start')

binz=[3628,3799,3887,4009,4114,4242,4370,4490,4605,4709,4857,4958,5152,5601]


binzz = [3312.3,\
3709.6,\
3843.125,\
3967.,\
4061.1,\
4210,\
4317.,\
4429.3,\
4562.55555556,\
4653.875,\
4789.22222222,\
4921.44444444,\
5045.9,\
5417.3,\
5698.25]

argon_detected = [\
2895,\
3017,\
3236,\
3274,\
3335,\
3362,\
3377,\
3448,\
3551,\
3628,\
3637,\
3641,\
3660,\
3676,\
3678,\
3698,\
3746,\
3772,\
3789,\
3799,\
3807,\
3809,\
3811,\
3829,\
3857,\
3868,\
3877,\
3887,\
3920,\
3928,\
3954,\
3956,\
3968,\
3971,\
3981,\
3983,\
4000,\
4009,\
4015,\
4017,\
4041,\
4043,\
4050,\
4066,\
4068,\
4094,\
4103,\
4114,\
4177,\
4183,\
4186,\
4197,\
4204,\
4220,\
4224,\
4233,\
4234,\
4242,\
4243,\
4266,\
4299,\
4309,\
4323,\
4325,\
4327,\
4341,\
4367,\
4370,\
4379,\
4381,\
4396,\
4403,\
4409,\
4450,\
4453,\
4462,\
4470,\
4490,\
4514,\
4517,\
4554,\
4557,\
4558,\
4569,\
4585,\
4604,\
4605,\
4617,\
4618,\
4629,\
4631,\
4638,\
4684,\
4705,\
4709,\
4734,\
4736,\
4751,\
4771,\
4776,\
4814,\
4828,\
4836,\
4857,\
4870,\
4884,\
4908,\
4921,\
4922,\
4934,\
4940,\
4956,\
4958,\
4972,\
4984,\
4988,\
5011,\
5019,\
5038,\
5046,\
5098,\
5151,\
5152,\
5304,\
5323,\
5349,\
5359,\
5371,\
5395,\
5436,\
5495,\
5540,\
5601,\
5608,\
5655,\
5671,\
5859]
for x in argon_detected:
    plt.axvline(x, alpha=.3, color='purple')
argon_detected = np.array(argon_detected)

for x in binzz:
    plt.axvline(x, alpha=1, color='red')

#plt.axvspan(2600,3000, alpha=0.4, color='#17A589', label=r'401,474 Pulses, 1 $Ar^{17+}$ 97 $Ar^{16+}$')
#plt.axvspan(3000,3400, alpha=0.4, color='#F39C12', label=r'128,516 Pulses, 6 $Ar^{17+}$ 606 $Ar^{16+}$')


#for x, color, label in zip(bins, colors, labels):
#    plt.axvspan(x[0],x[1], alpha=0.4, color=color, label=label)
#plt.legend(fontsize=9)


plt.show()