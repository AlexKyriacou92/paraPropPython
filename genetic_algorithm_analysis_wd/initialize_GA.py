import datetime
import os
import random
import h5py
import numpy as np
from scipy.interpolate import interp1d
import subprocess
import time

import sys
sys.path.append('../genetic_algorithm_analysis/')
from makeSim import createMatrix
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data, make_command_data, make_job
from selection_functions import selection

#=======================================================================================================================
# S
#=======================================================================================================================
