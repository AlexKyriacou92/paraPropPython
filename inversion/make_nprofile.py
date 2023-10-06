import os.path
import sys
import numpy as np
import time
import datetime
import h5py
import configparser
from matplotlib import pyplot as pl

from makeDepthScan import depth_scan_from_hdf
from util import do_interpolation_spline
sys.path.append('../')

import util
import paraPropPython as ppp
from receiver import receiver as rx
from transmitter import tx_signal
from data import create_sim, create_rxList_from_file, create_transmitter_array, create_hdf_FT
from data import create_tx_signal, bscan, bscan_rxList, create_hdf_bscan

fname_nprofile = 'share/guliya.txt'
nprof_decimate = util.get_profile_from_file_decimate(fname_nprofile, 0, 15, 0.5)
z_space = np.linspace(0, 15, len(nprof_decimate))
nprof_plot, zprof_plot = util.do_interpolation_spline(zprof_in=z_space, nprof_in=nprof_decimate, dz_out=0.05)

pl.plot(z_space, nprof_decimate)
pl.plot(zprof_plot, nprof_plot)
pl.show()

util.save_profile_to_txtfile(zprof=zprof_plot, nprof=nprof_plot, fname='share/guliya_spline.txt')