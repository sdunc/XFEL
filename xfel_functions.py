# Stephen Duncanson
# stephen.duncanson@uconn.edu
# Berrah Lab, proposal 2318
# Functions for the analysis of processed data

import numpy as np
import random
import scipy as sci
from scipy import signal
import matplotlib.pyplot as plt
import h5py as h5
from karabo_data import open_run, RunDirectory, H5File, by_index, by_id
import math
import time
from scipy.optimize import curve_fit
import os
from mpl_toolkits import mplot3d


def open_processed_run(run_number):
    '''
    Returns the h5py file object for a processed run
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


def get_tof(charge, mass=40):
    '''
    The result of the linear calibration (that I might need to redo)
    accepts the charge [2,17] as well as mass (defaults to Argon 40)
    and returns tof in ns/2
    '''
    slope = 6685.782541703501
    intercept = 269.49516087660595
    return slope*np.sqrt(mass/charge)+intercept


def get_ratio_error(u, sigma_u, v, sigma_v, x):
    '''
    returns the error in the ratio between the pulses
    See bevington pg 44.
    '''
    return np.sqrt((((x**2)*(sigma_u**2))/(u**2))+(((x**2)*(sigma_v**2))/(v**2)))


def round_intensity_to_closest_bin(n, bins):
    '''
    takes bins, the array of bins to round to
    and n, the number we are placing in the closest bin
    '''
    bins.sort()
    bin_min = min(bins)
    bin_max = max(bins)
    bin_size = int(bins[1])-int(bins[0])
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
    Edwin's Method for getting the pulse time
    not sure where he got these hardcoded values
    time unit: ns/2
    '''
    t_min_period = 1760         # t_min_period is when pulses are in consecutive bunches. unit: ns/2
    every_x_pulse_in_train = 24 # no units, an amount
    return t_min_period*every_x_pulse_in_train


def get_peaks_at_int(run_number_array, bins):
    '''
    return a dictionary of all the peaks found at various bins of intensity
    peaks_at_int[intensity] = ([all peaks], # pulses)
    Use a core python list here for the v since I'll be appending a lot.
    '''
    # default bins? maybe think of a better way to get the default
    peaks_at_int = {}
    # add all the bins (in order) to the dict 
    for run in run_number_array:
        f = open_processed_run(run)
        number_of_peaks_per_pulse = np.array(f['num_peaks_per_pulse'])
        intensity_per_pulse = np.array(f['intensity_per_pulse'])
        all_peaks = np.array(f['all_peaks'])
        f.close()
        # the indicies to slice along to seperate all peaks into peaks per pulse
        # is the cumulative sum of peaks_per_pulse
        peak_indicies = np.cumsum(number_of_peaks_per_pulse)
        # spit all_peaks along those indicies to get peaks on a pulse by pulse basis
        peaks_per_pulse = np.array(np.array_split(all_peaks, peak_indicies))[0:-1]
        for i, p in zip(intensity_per_pulse, peaks_per_pulse):
            if int(i) in peaks_at_int:
                peaks_at_int[i][0].extend(peaks_per_pulse)
                peaks_at_int[i][1]+=1
            else:
                # the intensity is not yet in the dict.
                peaks_at_int[int(i)] = [[peaks_per_pulse], 1]
    return peaks_at_int


def get_count_mode_at_intensity(peaks_at_int):
    '''
    takes the dictionary of peaks at intensity and returns 
    a dictionary of the count mode at each intensity
    '''
    pulse_time = get_pulse_time()
    bin_size = 1 # ns/2 
    count_mode_at_intensity = {}
    for intensity in peaks_at_int:
        all_peaks = peaks_at_int[intensity][0]
        count_at_intensity = np.histogram(all_peaks, bins=range(0, pulse_time+1, bin_size))[0]
        count_mode_at_intensity[intensity] = (count_at_intensity, peaks_at_int[intensity][1])
    return count_mode_at_intensity


def get_i_peaks_at_int(run_number_array):
    '''
    all peaks for integer bins
    '''
    peaks_at_int = {}
    for run in run_number_array:
        f = open_processed_run(run)
        number_of_peaks_per_pulse = np.array(f['num_peaks_per_pulse'])
        intensity_per_pulse = np.array(f['intensity_per_pulse'])
        all_peaks = np.array(f['all_peaks'])
        f.close()
        # the indicies to slice along to seperate all peaks into peaks per pulse
        # is the cumulative sum of peaks_per_pulse
        peak_indicies = np.cumsum(number_of_peaks_per_pulse)
        # spit all_peaks along those indicies to get peaks on a pulse by pulse basis
        peaks_per_pulse = np.array(np.array_split(all_peaks, peak_indicies))[0:-1]
        for i, p in zip(intensity_per_pulse, peaks_per_pulse):
            # i: intensity of the pulse
            # p: peaks in that pulse
            peaks_in_pulse = list(p)
            integer_intensity = int(i)
            if integer_intensity in peaks_at_int:
                peaks_at_int[integer_intensity][0].extend(peaks_in_pulse)
                peaks_at_int[integer_intensity][1] = int(peaks_at_int[integer_intensity][1])+1
            else:
                peaks_at_int[integer_intensity] = [peaks_in_pulse,1]
    return peaks_at_int


def i_get_count_mode_at_intensity(peaks_at_int):
    '''
    FOR THE MESH
    normalized count mode for ever integer value of intensity
    a dictionary of the count mode at each intensity
    '''
    pulse_time = get_pulse_time()
    bin_size = 1 # ns/2 
    count_mode_at_intensity = {}
    for intensity in peaks_at_int:
        all_peaks = peaks_at_int[intensity][0]
        count_at_intensity = np.histogram(all_peaks, bins=range(0, pulse_time+1, bin_size))[0]
        count_mode_at_intensity[intensity] = count_at_intensity#/#(peaks_at_int[intensity][1])
    return count_mode_at_intensity 


def fill_i_count(count_mode_at_intensity):
    ''' fill in all the 'empty' rows where no int was recorded'''
    max_intensity = sorted(count_mode_at_intensity)[-1]
    min_intensity = sorted(count_mode_at_intensity)[0]
    ints = range(min_intensity, max_intensity, 1)
    for i in ints:
        if i not in count_mode_at_intensity:
            count_mode_at_intensity[i] = np.zeros(42240)

    return count_mode_at_intensity


def get_ion_count_at_intensity(count_mode_at_intensity):
    '''
    count the number of each species of ion at each bin of intensity
    normalized to the 'intensity' (why is everything called intensity?) 
    by dividing by the number of pulses
    build a 2D array to store all the ion counts at each intensity
    shape (bins, 18)
    each row: [intensity, 0, ar+1count, ar+2count,...,ar+17count]
    '''
    epsilon = 15
    # create the empty 2d array, number of bins (rows) = number of keys in count_mode_at_intensity
    # 18 columns because 0 = int, 1-17 = count of ion
    ion_count_grid = np.empty((len(count_mode_at_intensity),18),dtype=float)
    counter = 0
    for intensity in count_mode_at_intensity:
        row = np.empty(18,dtype=float)
        count = count_mode_at_intensity[intensity][0]
        number_of_pulses = count_mode_at_intensity[intensity][1]
        row[0] = intensity
        row[1] = 0 # since we assume there are no +1 cations
        row[2]= np.sum(count[int(get_tof(2)-epsilon):int(get_tof(2)+epsilon)])/number_of_pulses
        row[3]= np.sum(count[int(get_tof(3)-epsilon):int(get_tof(3)+epsilon)])/number_of_pulses
        row[4]= np.sum(count[int(get_tof(4)-epsilon):int(get_tof(4)+epsilon)])/number_of_pulses
        row[5]= np.sum(count[int(get_tof(5)-epsilon):int(get_tof(5)+epsilon)])/number_of_pulses
        row[6]= np.sum(count[int(get_tof(6)-epsilon):int(get_tof(6)+epsilon)])/number_of_pulses
        row[7]= np.sum(count[int(get_tof(7)-epsilon):int(get_tof(7)+epsilon)])/number_of_pulses
        row[8]= np.sum(count[int(get_tof(8)-epsilon):int(get_tof(8)+epsilon)])/number_of_pulses
        row[9]= np.sum(count[int(get_tof(9)-epsilon):int(get_tof(9)+epsilon)])/number_of_pulses
        row[10]= np.sum(count[int(get_tof(10)-epsilon):int(get_tof(10)+epsilon)])/number_of_pulses
        row[11]= np.sum(count[int(get_tof(11)-epsilon):int(get_tof(11)+epsilon)])/number_of_pulses
        row[12]= np.sum(count[int(get_tof(12)-epsilon):int(get_tof(12)+epsilon)])/number_of_pulses
        row[13]= np.sum(count[int(get_tof(13)-epsilon):int(get_tof(13)+epsilon)])/number_of_pulses
        row[14]= np.sum(count[int(get_tof(14)-epsilon):int(get_tof(14)+epsilon)])/number_of_pulses
        row[15]= np.sum(count[int(get_tof(15)-epsilon):int(get_tof(15)+epsilon)])/number_of_pulses
        row[16]= np.sum(count[int(get_tof(16)-epsilon):int(get_tof(16)+epsilon)])/number_of_pulses
        row[17]= np.sum(count[int(get_tof(17)-epsilon):int(get_tof(17)+epsilon)])/number_of_pulses
        ion_count_grid[counter] = row
        counter+=1
    return ion_count_grid


