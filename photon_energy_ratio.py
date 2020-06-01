# stephen duncanson
# photon energy ratio

import numpy as np
import xfel_functions as xf
import matplotlib.pyplot as plt

runs_to_analyze = [29,34,35,36,37,38,39,44,45,46,47,48,52,53,55,57,58,59,71,73,93]
bin_size = 10

xf.get_peaks_at_photon_energy(runs_to_analyze)