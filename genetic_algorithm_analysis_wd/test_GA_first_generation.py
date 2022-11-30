import datetime
import os
import random

import h5py
import numpy as np
from scipy.interpolate import interp1d

from makeSim import createMatrix

import sys
sys.path.append('../genetic_algorithm_analysis/')
from genetic_functions import initialize_from_analytical, roulette, initalize_from_fluctuations
from pleiades_scripting import make_command, test_job, submit_job, test_job_data
from selection_functions import selection
import subprocess
import time

nStart = 10000
#To start with -> just run 15 individuals
nIndividuals = 100
nGens = 10

fname_start = 'start_profiles/aletsch_glacier_2.txt'