import numpy as np
from permittivity import *
import h5py
import sys
import datetime
import os

fname_out = str(sys.argv[1])
output_h5 = h5py.File(fname_out + '.h5', 'w')
output_npy = fname_out + '.npy'
output_h5.create_dateset('receiverData', data=output_npy)

simul_end = datetime.datetime.now()
output_h5.attrs["EndTime"] = simul_end.strftime("%Y\%m\%d %H:%M:%S") #time that simulation ends

#os.remove(output_npy)
output_h5.close()
