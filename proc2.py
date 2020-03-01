# Stephen Duncanson
# stephen.duncanson@uconn.edu
# 2/14/20, 2/17/20
# Function to process raw data obtained from beamtime at European XFEL
# Berrah Lab

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

def get_tof(charge):
    mass = 40
    slope = 6685.782541703501
    #6682.717635120452
    # old slopes
    #6546.308107835526
    #6471.270479910258#6449.381322769152
    intercept = 269.49516087660595
    #272.8157786973061
    # old intercepts
    #788.8355991091994
    #1127.6940289757622#1248.6960382392426
    return slope*np.sqrt(mass/charge)+intercept

def normalize_wrapped_trace(wrapped_trace):
    '''
    Inputs:
        wrapped_trace: wrapped raw trace
    Outputs:
        normalized_trace: normalized signal between 0-1
    Todo:
        Units: Figure out what the units of the normalized trace are
        Fix offset: center properly at 0
    Note:
        Normalizing makes it look/feel a lot nicer but is it?
    '''
    max_val = max(wrapped_trace)
    min_val = min(wrapped_trace)
    denomin = max_val - min_val
    return [(x-min_val)/denomin for x in wrapped_trace]


def wrap_raw_trace(raw_trace, pulses_per_train, pulse_time):
    '''
    Inputs:
        raw_trace:          The raw trace from this instrument:
                            SQS_DIGITIZER_UTC1/ADC/1:network', 'digitizers.channel_1_A.raw.samples
        pulses_per_train:   Number of pulses per train. 28 for this proposal
        pulse_time:         The time in ns/2 of one pulse
    Outputs:
        wrapped_trace: The raw trace wrapped around the pulse time
    Todo:
        Fix units: ns/2 -> ns
        Possible speed improvement using list comprehension instead of for^2 loops
    Note:
        Wrapping a trace is when there could be peaks in the trace at the wrong time
        and so we wrap it along a domain of times to see where the peak really belongs
    '''
    for i in range(1, pulses_per_train-1):
        for k in range(pulse_time): 
            # sum the 2D array along an axis of all integer multiples of pulses
            raw_trace[k] += raw_trace[k+i*pulse_time]
    wrapped_trace = raw_trace[0:pulse_time-1]
    return wrapped_trace


def save_reduced_run(intensity_per_pulse, tof_count_per_pulse, tof_spec, run_number, normalized_sum_trace):
    '''
    Inputs:
        intensity_per_pulse: 1D flattened array of the intensity per pulse
        tof_count_per_pulse: 2D Array of the count mode on a pulse by pulse basis. shape: (bins, pulses)
        tof_spec: The bin edges or xcoords of the tof count mode spectrum
        run_number: The run number, [1,128] used to define the savefile name
        normalized_sum_trace: output of normalize_wrapped_trace for all the traces in the run (sum_trace)
    Outputs:
        run_RUN_NUMBER_proc2.h5: an HDF5 file containing all the input arrays in the local dir
    Todo:
        Add more reduced pieces of data as needed!
    Note:
        Perhaps I should add groups?
        If I have time I could read more h5py documentation
    '''
    base_filename = "run_"+str(run_number)+"_proc2.h5"
    f5 = h5.File(base_filename,'w')    
    print("Saving list to file: ",base_filename)
    hist = np.array(tof_count_per_pulse)
    try:
        f5.create_dataset("tof_count_per_pulse", data=hist)
    except:
        f5.close()  
    try:
        f5.create_dataset("normalized_sum_trace", data=normalized_sum_trace)
    except:
        f5.close()
    try:
        f5.create_dataset("tof_spec", data=tof_spec)
    except:
        f5.close()
    try:
        f5.create_dataset("intensity_per_pulse",data=intensity_per_pulse,dtype='float32')
    except:
        f5.close()
    f5.close()
    
    
def get_data(f):
    '''
    Input:
        f: h5 file object made using save_reduced_run and opened using open_processed_run
    Output:
        intensity_per_pulse: 1D flattened array of the intensity per pulse
        tof_count_per_pulse: 2D Array of the count mode on a pulse by pulse basis. shape: (bins, pulses)
        tof_spec: The bin edges or xcoords of the tof count mode spectrum
        normalized_sum_trace: output of normalize_wrapped_trace for all the traces in the run (sum_trace)
    Todo:
        N/A?
    Note: 
        Keep updated with other save/open functions if changed
    '''
    try:
        intensity_per_pulse = np.array(f['intensity_per_pulse'])
    except:
        intensity_per_pulse = None
        print("Error, file does not contain intensity_per_pulse!")
    try: 
        tof_spec = np.array(f['tof_spec'])
    except:
        tof_spec = None
        print("Error, file does not contain tof_spec!")
    try:
        tof_count_per_pulse = np.array(f['tof_count_per_pulse'])
    except:
        tof_count_per_pulse = None
        print("Error, file does not contain tof_count_per_pulse!") 
    try:
        normalized_sum_trace = np.array(f['normalized_sum_trace'])
    except:
        normalized_sum_trace = None
        print("Error, file does not contain normalized_sum_trace!")
    return intensity_per_pulse, tof_count_per_pulse, tof_spec, normalized_sum_trace


def open_processed_run(run_number):
    '''
    Inputs:
        run_number: the number of the reduced run to open, made with process_raw_run
    Outputs:
        f: h5 file object, should be loaded into get_data
    Todo:
        better dir support?
    Note:
        analouge of open_run but for processed data
        also changed, see the _proc2.h5, this one is for the files created by scripts in proc2
        depending on the run_number parameter which gets passed, it will go to the expected folder in
        /Stephen/(number)/open the .h5 file
        this uses the relative path, so expects to be run within the same directory as the folders 
    '''
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    try:
        f = h5.File(target_file, 'r')
    except:
        print("No processed run file found! for run: "+str(run_number))
        return

    return f

def save_intensity(run_number):
    '''
    Inputs:
        run_number: just the number of which run we are reducing
                    see comment on process_raw_run for more info
    Output:
        proc file is created if it does not exist, if it exists, the int is added.
    Todo:
        idk
    Note:
        This is part of me splitting up the process_raw_run 
        to get around my time restrictions on the cluster
    '''
    proposal_number = 2318      # The proposal number for this beamtime
    try:
        r = open_run(proposal=proposal_number, run=run_number)
        print("Opened run: "+str(run_number)+"\tfor proposal: "+str(proposal_number))
    except:
        print("Run: "+str(run_number)+" not found, exiting")
        return
    
    train_IDs = r.train_ids
    number_of_trains = len(train_IDs) 
    print(number_of_trains)
    
    # determine the pulses per train
    intensities_0 = r.get_array('SA3_XTD10_XGM/XGM/DOOCS:output','data.intensitySa3TD')[0]
    for i in range(len(intensities_0)):
        if intensities_0[i] == 1.0:
            pulses_per_train = i
            print("Pulses per train: "+str(pulses_per_train))
            break
    
    # 2D inensity list
    # region of interest is the each pulse x each train, which is also shape of array
    # get the intensity associated with each pulse within each train
    # xgm is X-Ray Gas Monitor
    # xgm has 1. Two beam intensity monitors to measure intensity along both x and y direction.
    # and 2. Two beam position monitors to determine x and y beam positions.
    # xgm is a 2d array with the shape (n trains, 1000)
    intensity_list = np.array(r.get_array('SA3_XTD10_XGM/XGM/DOOCS:output','data.intensitySa3TD',roi=by_index[0:pulses_per_train])[0:number_of_trains])
    # and flatten the intensity list into a 1D array, so it has the same shape as peaks_in_pulse 
    # that is, total number of pulses

    intensity_per_pulse = intensity_list.flatten()
    print(max(intensity_per_pulse))
    print(min(intensity_per_pulse))
    plt.hist(intensity_per_pulse, bins=500)
    plt.show()
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    f = h5.File(target_file, 'a')
    try:
        f.create_dataset("intensity_per_pulse", data=intensity_per_pulse)
        f.close()
    except:
        print("Int already saved!")
        f.close()
        return
    return


def save_gatt(run_number):
    '''
    Inputs:
        run_number: just the number of which run we are reducing
                    see comment on process_raw_run for more info
    Output:
        proc file is created if it does not exist, if it exists, the gatt is added.
    Todo:
        idk
    Note:
        This is part of me splitting up the process_raw_run 
        to get around my time restrictions on the cluster
    '''
    # gatt is Gas Attenuator (arrangement of resistors reduces the strength of a signal.)
    # I think it normalizes a signal to a (0,100) range
    gatt = np.array(r.get_array('SA3_XTD10_GATT/MDL/GATT_TRANSMISSION_MONITOR', 'Estimated_Tr.value'))/100
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    f = h5.File(target_file, 'a')
    f.create_dataset("gatt", data=gatt)
    f.close()      
    return


def save_photon_energy(run_number):
    '''
    WIP
    '''
    proposal_number = 2318      # The proposal number for this beamtime
    try:
        r = open_run(proposal=proposal_number, run=run_number)
        print("Opened run: "+str(run_number)+"\tfor proposal: "+str(proposal_number))
    except:
        print("Run: "+str(run_number)+" not found, exiting")
        return
    # get the photon energy of the run
    # units are eV, rounded to 3 decimal places
    # the Germans had a try except for this, I could add that...
    #hv = round(float(r.get_array('SA3_XTD10_UND/DOOCS/PHOTON_ENERGY', 'actualPosition.value')[0]), 3)
    #print("Photon Energy for run "+str(run_number)+": "+str(hv)+ "eV")
    
    hv = r.get_array('SA3_XTD10_UND/DOOCS/PHOTON_ENERGY', 'actualPosition.value')
    plt.plot(hv)
    plt.show()
    #print(hv)
    return

def save_nws_trace(run_number):
    '''
    save normalized, wrapped, summed trace
    Should work. 
    '''
    proposal_number = 2318      # The proposal number for this beamtime
    try:
        r = open_run(proposal=proposal_number, run=run_number)
        print("Opened run: "+str(run_number)+"\tfor proposal: "+str(proposal_number))
    except:
        print("Run: "+str(run_number)+" not found, exiting")
        return
    t_zero = 18000              # train time offset unit: ns/2
    chunk_size = 1

    train_IDs = r.train_ids
    number_of_trains = len(train_IDs) 
    # determine the pulses per train
    intensities_0 = r.get_array('SA3_XTD10_XGM/XGM/DOOCS:output','data.intensitySa3TD')[0]
    for i in range(len(intensities_0)):
        if intensities_0[i] == 1.0:
            pulses_per_train = i
            print("Pulses per train: "+str(pulses_per_train))
            break
    pulse_time = 24*1760
    train_time = pulse_time * pulses_per_train
    sum_trace = 0
    for i in range(number_of_trains):
        i_shift = math.floor(i/chunk_size)*chunk_size
        sel = r.select_trains(by_index[i:i+chunk_size]) 
        trace_by_train = np.array(sel.get_array('SQS_DIGITIZER_UTC1/ADC/1:network', 'digitizers.channel_1_A.raw.samples',None,roi=by_index[t_zero:t_zero+train_time]),dtype=np.float64)[0]
        trace_by_train*=(-1)
        wrapped_trace = wrap_raw_trace(trace_by_train, pulses_per_train, pulse_time)
        sum_trace += wrapped_trace
        #plt.plot(sum_trace)
        #plt.show()
        #plt.pause(0.005)
    normalized_sum_trace = normalize_wrapped_trace(sum_trace)
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    f = h5.File(target_file, 'a')
    f.create_dataset("normalized_sum_trace", data=normalized_sum_trace)
    f.close()  
    
    
def save_tof_time_axis(run_number):
    pulse_time = 24*1760
    bin_size = 1
    tof_spec = range(0, pulse_time-1, bin_size)
    tof_spec = tof_spec[0:-1] # to match the shape of the hist I will create
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    f = h5.File(target_file, 'a')
    f.create_dataset("tof_time_axis", data=tof_spec)
    f.close()
    return


def save_peaks_per_pulse(run_number, starting_index, trains):
    '''    
    
    '''
    proposal_number = 2318      # The proposal number for this beamtime
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    prominence = 100            # Prominence is an arbitrary threshhold used to select peaks. Edwin used 100.
    wlen = 20                   # 'wlen' or window length helps speed up peak finding algorithm, see above link.
    # A run is composed of many(!) trains
    # a train is composed of many pulses
    # a pulse is composed of many photons
    every_x_pulse_in_train = 24 # no units, an amount
    t_min_period = 1760         # t_min_period is when pulses are in consecutive bunches. unit: ns/2
    t_zero = 18000              # train time offset unit: ns/2
    bin_size = 1                # unit for tof spectrum unit: ns/2
    pulse_time = every_x_pulse_in_train * t_min_period  # time for one pulse unit: ns/2

    try:
        r = open_run(proposal=proposal_number, run=run_number)
        print("Opened run: "+str(run_number)+"\tfor proposal: "+str(proposal_number))
    except:
        print("Run: "+str(run_number)+" not found, exiting")
        return
    
    local_dir = os.path.dirname('__file__')
    target_file = os.path.join(local_dir,'run_'+str(run_number)+'_proc2.h5')
    
    # get all IDs in train
    train_IDs = r.train_ids
    number_of_trains = len(train_IDs) 
    print(number_of_trains)
    
    # determine # of pulses per train, break loop once found
    intensities_0 = r.get_array('SA3_XTD10_XGM/XGM/DOOCS:output','data.intensitySa3TD')[0]
    for i in range(len(intensities_0)):
        if intensities_0[i] == 1.0:
            pulses_per_train = i
            print("Pulses per train: "+str(pulses_per_train))
            break

    # calculate constants for tof spectrum
    train_time = pulse_time * pulses_per_train          # time for the train unit: ns/2
    number_of_pulses = number_of_trains * pulses_per_train # number of pulses in run

    # create empty array
    tof_count_per_pulse = []
  
    # how many traces (trains) to load into data at once
    # right now this code will only work with chunk_size = 1
    # which is really slow, but is needed for ions per pulse!
    # see Todo^
    chunk_size = 1
    f = h5.File(target_file, 'a')
    try:
        # create a group for all pulses
        f.create_group('/tof_peaks_per_pulse')#, (number_of_pulses,range(0, pulse_time-1, bin_size)), dtype='int8', maxshape=(None,))
        f.close()
    except:
        print("group already made!")
        f.close()
    # iterate over each train in the run to avoid overallocating memory
    # used for naming! dont worry lol
    max_zeros = (len(str(number_of_trains)))*"0"

    for i in range(starting_index, starting_index+trains):
        if i > number_of_trains:
            # all done
            return
        else:
            i_shift = math.floor(i/chunk_size)*chunk_size
            sel = r.select_trains(by_index[i:i+chunk_size]) 
            trace_by_train = np.array(sel.get_array('SQS_DIGITIZER_UTC1/ADC/1:network', 'digitizers.channel_1_A.raw.samples',None,roi=by_index[t_zero:t_zero+train_time]),dtype=np.float64)[0]
            trace_by_train*=(-1)

            peaks = sci.signal.find_peaks(trace_by_train, prominence=prominence, wlen=wlen)
            # peaks[0]: indicies in single_train_traces of peaks!
            # peaks[1]: dictionary of properties

            # find out which pulse the peaks belongs to
            # divide the index within trace where a peak was found by the time of each pulse
            # pulse_of_peaks will look like [1,1,1,2,2,3,5,6,...]
            # an array with the same shape as peaks[0]
            # it takes the peaks found and generates an array of the same shape
            # except each index lists the pulse which that peak belongs to
            # t_min_period
            pulse_of_peaks = list(np.floor(peaks[0]/pulse_time).astype(int))

            # get the number of peaks per pulse
            # this many element will need to be removed from peaks[0]
            # and evaulated as that pulse
            n_peaks_per_pulse = np.bincount(pulse_of_peaks, minlength=pulses_per_train)


            # Wrap the time of flight of peaks[0]
            # create empty array same shape as all_peaks
            t_peaks = np.empty_like(peaks[0])
            # populate array with the modulus (remainder) of all_peaks and pulse_time
            # each element of all_peaks is the index within the single_train_traces which it came out of
            # that represents a 'peak' of the data
            # I am guessing that if the data is collected at a certain rate we can interact with it 
            # in this way
            t_peaks = np.mod(peaks[0], pulse_time)
            # make all_peaks a list
            all_peaks = t_peaks.tolist()

            # next I need to split this up to a pulse by pulse basis
            f = h5.File(target_file, 'a')
            counter = 0
            for peak_count in n_peaks_per_pulse:
                abs_pulse_num = pulses_per_train*i + counter
                peaks_in_pulse = []
                for p in range(peak_count):
                    # get the 0th element p times
                    peaks_in_pulse.append(all_peaks.pop(0))
                peaks_in_pulse = np.array(peaks_in_pulse)
                group = f.get('tof_peaks_per_pulse')
                counter+=1
                # how many places of 0s total - how long the abs pulse num (pulse id) is
                zeros = len(str(number_of_trains)) - len(str(abs_pulse_num))
                added_zeros = "0"*zeros
                dataset_name = "pulse_"+str(added_zeros)+str(abs_pulse_num)
                try:
                    # save as unsigned int 16 to save space! 
                    group.create_dataset(str(dataset_name),data=peaks_in_pulse,dtype='uint16')
                except:
                    print("Error! that pulse is already saved, check parameters.")
                    print("i=", i)
                    pass
            f.close()
    return


def reduce(run_number):
    
    
    
    pass