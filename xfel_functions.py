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
    print("Opening file: "+str(target_file))
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


def get_bins(intensity_range, bin_size):
    '''
    take the two endpoints found from get_intensity_range() 
    and use them to generate an array of bins
    '''
    lower_bound = intensity_range[0]
    a = (lower_bound // bin_size) * bin_size
    # Larger multiple 
    b = a + bin_size
    lower_bound = int(b if lower_bound - a > b - lower_bound else a)
    upper_bound = intensity_range[1]
    a = (upper_bound // bin_size) * bin_size
    # Larger multiple 
    b = a + bin_size
    upper_bound = int(b if upper_bound - a > b - upper_bound else a)
    bins = []
    for i in range(lower_bound, upper_bound+bin_size, bin_size):
        bins.append(i)
    return bins   

def get_pulse_time():
    '''
    Edwin's Method for getting the pulse time
    not sure where he got these hardcoded values
    time unit: ns/2
    '''
    t_min_period = 1760         # t_min_period is when pulses are in consecutive bunches. unit: ns/2
    every_x_pulse_in_train = 24 # no units, an amount
    return t_min_period*every_x_pulse_in_train


def get_max_intensity(run_number_array):
    '''
    find the maximum pulse intensity across all runs_to_analyze
    updated to not presuppose a certain range of pulse intensities
    '''
    first_run_number = run_number_array.pop()
    first_run = open_processed_run(first_run_number)
    global_max_int = int(np.amax(np.array(f['intensity_per_pulse'])))
    for run in run_number_array:
        f = open_processed_run(run)
        intensity_per_pulse = np.array(f['intensity_per_pulse'])
        f.close()
        local_max = int(np.amax(intensity_per_pulse))
        if local_max >= global_max_int:
            global_max_int = local_max
    return global_max_int


def get_intensity_range(run_number_array):
    '''
    find the maximum/minimum pulse intensity across all runs_to_analyze
    average all max/min pulse intensities to get average endpoints
    averge endpoints become the intensity range for the 3d plot
    make sure no negative intensities pollute the min average
    this will crash if run_number_array is of length 1
    or not an array
    '''
    max_ints = []
    min_ints = []
    for run in run_number_array:
        f = open_processed_run(run)
        intensity_per_pulse = np.array(f['intensity_per_pulse'])
        f.close()
        max_ints.append(int(np.amax(intensity_per_pulse)))
        min_ints.append(int(np.amin(intensity_per_pulse)))
    # to ensure that the endpoints are not messed up by negative/weird intensity
    # resulting from instrumentation or recording errors (of which I have noticed a couple)
    # lets clean up the lists before finding the averages
    for x in min_ints:
        if x <= 0:
            min_ints.remove(x)
    for x in max_ints:
        if x <= 0:
            max_ints.remove(x)
    mean_min = int(np.mean(min_ints))
    mean_max = int(np.mean(max_ints))
    return [mean_min, mean_max]

def build_peaks_at_int_dict(max_intensity):
    '''
    create the dictionary: peaks_at_int
    find max and min intensity across all runs
    for every integer in range [min, max] create a key in the dict
    peaks_at_int[intensity] = [[],0] #empty list and 0 pulses at int
    return this and pass to get_peaks_at_int
    that way it will not have to check and make bins and all that garbage
    '''
    peaks_at_int = {}
    # for all integers in range [min, max], add a key to the dictionary peaks_at_int
    # starting at 0 because of erroreous vaues of intensity being negative
    for i in range(0, max_intensity+1, 1):
        peaks_at_int[i] = [[],0]
    return peaks_at_int


def get_all_peaks(run_number_array):
    peaks = []
    for run in run_number_array:
        f = open_processed_run(run)
        all_peaks = np.array(f['all_peaks'])
        peaks.extend(all_peaks)
        f.close()
    return peaks

def get_all_count(all_peaks):
    bin_size = 1
    pulse_time = get_pulse_time()
    all_count = np.histogram(all_peaks, bins=range(0, pulse_time+1, bin_size))[0]
    return all_count

def get_peaks_at_photon_energy(run_number_array):
    get_peaks_at_photon_energy = {}
    for run in run_number_array:
        f = open_processed_run(run)
        number_of_peaks_per_pulse = np.array(f['num_peaks_per_pulse'])
        all_peaks = np.array(f['all_peaks'])
        photon_energy = np.array(f['photon_energy'])
        print(photon_energy)
        f.close()        


def get_peaks_at_int(run_number_array, bins):
    '''
    return a dictionary of all the peaks found at various bins of intensity
    peaks_at_int[intensity] = ([all peaks], # pulses)
    '''
    peaks_at_int = {}
    # fill the dictionary with intensities from the bins array
    for bin in bins:
        peaks_at_int[bin] = [[], 0]
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
            if i < (bins[1]-(bins[1]-bins[0])):
                # pulse energy is below 0, dont include
                # changed this so that if it is less than 
                # 1 bin width away that it will not include
                # 300 - (600-300) example of that line
                pass
            elif i > (bins[-1]+(bins[1]-bins[0])):
                # dont include this, the pulse energy is too big
                # greater that 1 bin width beyond the cutoff
                pass
            else:
                # convert the raw intensity, i, to the closest bin
                bin_int = round_intensity_to_closest_bin(i, bins)
                peaks_at_int[bin_int][0].extend(p)
                peaks_at_int[bin_int][1]+=1
    return peaks_at_int


def crude_interp_fill_dict(peaks_at_int, max_intensity):
    prev_val = [[],0]
    for i in range(1,max_intensity):
        if i in peaks_at_int:
            prev_val = peaks_at_int[i]
        else:
            peaks_at_int[i] = prev_val
    return peaks_at_int


def get_count_per_pulse_at_intensity(peaks_at_int):
    '''
    takes the dictionary of peaks at intensity and returns 
    a dictionary of the count mode at each intensity
    '''
    print("Pulse Energy",end='\t')
    print("Ar17+ Counts",end='\t')
    print('Number of pulses')
    pulse_time = get_pulse_time()
    bin_size = 1 # 5 ns, 1 = ns/2
    # want to use a bigger bin to make the shape a little more square
    #print(pulse_time+1)
    # are the zeros the same shape?
    count_at_intensity = {}
    for intensity_bin in peaks_at_int:
        # check if there are 0 pulses
        if peaks_at_int[intensity_bin][0] == []:
            # there are no peaks at this intensity (even if there are pulses!)
            count_at_intensity[intensity_bin] = np.zeros(pulse_time)
        else: 
            # there are peaks at this intensity
            all_peaks = peaks_at_int[intensity_bin][0]
            count = np.histogram(all_peaks, bins=range(0, pulse_time+1, bin_size))[0]
            # divide by number of pulses
            #print(intensity_bin, end='\t')
            #ar17_count = np.sum(count[int(get_tof(17)-12):int(get_tof(17)+12)])
            #print(ar17_count,end='\t')
            #print(peaks_at_int[intensity_bin][1])
            count = count/peaks_at_int[intensity_bin][1]
            #print(intensity_bin, end='\t')
            #print(peaks_at_int[intensity_bin][1])
            count_at_intensity[intensity_bin] = count
    return count_at_intensity


def mush_tof(square_counts, divisions=2640):
    '''
    '''
    cols = []
    split_array = np.split(square_counts, divisions, axis=1)
    for sub_array in split_array:
        sa = np.sum(sub_array, axis=1)
        cols.append(sa)
    cols = tuple(cols)
    combined_array = np.column_stack(cols)
    return combined_array


def nearest_neighbor_interpolation(square_counts):
    '''
    '''
    # split the square array on a column by column basis
    # taking slices at various times of flight as intensity varies
    number_of_columns = np.shape(square_counts)[1] # (rows, columns)
    number_of_rows = np.shape(square_counts)[0]
    split_array = np.split(square_counts, number_of_columns, axis=1)
    # use list bc its easier ;)
    compare_arr = list(np.zeros(number_of_rows))
    split_array = [list(a) for a in split_array]
    for tof_slice in split_array:
        if tof_slice == compare_arr:
            # if all zeros, skip it! waste of time.
            #print("Skipping, waste of time")
            pass
        else:
            #print("Not all zeros")
            for count in tof_slice:
                if tof_slice.index(count) == 0:
                    # first element in the list, special case
                    if count == 0:
                        # if the count is zero
                        # we interpolate
                        # half of nearest neighbor
                        tof_slice[0] = tof_slice[1]/2
                if tof_slice.index(count) == len(tof_slice)-1:
                    # last elemt in the list, special case
                    if count == 0:
                        # if 0 we interpolate
                        tof_slice[-1] = tof_slice[-2]/2
                else:
                    if count == 0:
                        # not an extreme case, 2 neighbors
                        # also 0, interpolate
                        index = tof_slice.index(count)
                        next_val = tof_slice[index+1]
                        prev_val = tof_slice[index-1]
                        average = (prev_val+next_val)/2
                        # set to average, hope that works
                        tof_slice[index] = average
                    else:
                        # not extreme, has value, pass
                        pass
    cols = tuple(split_array)
    combined_array = np.column_stack(cols)
    return combined_array


# little snippit to save dict
#import csv
#w = csv.writer(open("output.csv", "w"))
#for key, val in peaks_at_int.items():
#    w.writerow([key, val])

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


