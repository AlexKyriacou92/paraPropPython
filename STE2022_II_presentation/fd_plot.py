import numpy as np
import sys
sys.path.append('../')

import paraPropPython as ppp
from receiver import receiver
from data import bscan

import scipy.signal as signal
import util
import h5py
import matplotlib.pyplot as pl
import peakutils as pku
import sys


iceLength = 60
iceDepth = 15.0
dx = 0.2
dz = 0.02
airHeight = 5.0

freq_1 = 1.3
freq_2 = 1.8
freq = freq_1

band = 0.2

sim = ppp.paraProp(iceLength=iceLength, iceDepth=iceDepth, dx=dx, dz=dz, airHeight=airHeight)

fname = 'share/aletsch/aletsch_0604_max_peak_15m.txt'
nprof_data = np.genfromtxt(fname)
z_profile = nprof_data[:,0]
n_profile = nprof_data[:,1]
sourceDepth = 14

sim.set_n(zVec=z_profile, nVec=n_profile)
sim.set_dipole_source_profile(freq, sourceDepth)
sim.set_cw_source_signal(freq)
sim.do_solver()

x = sim.get_x()
z = sim.get_z()

absu = abs(np.transpose(sim.get_field()))

fig = pl.figure(figsize=(12,12),dpi=120)
ax1 = pl.subplot2grid((3,2), (0,0),colspan=3)
pmesh = ax1.imshow(20*np.log10(absu), aspect='auto', cmap='hot', vmin=-110, vmax=20, extent=(x[0], x[-1], z[-1], z[0]))
cbar = pl.colorbar(pmesh)
cbar.set_label('Power [dBu]')
ax1.set_title('Absolute Field, f = ' + str(freq*1e3) + ' MHz')
ax1.fill_betweenx(z, 10-0.04, 10+0.04, where=z>0,color='k')
ax1.fill_betweenx(z, 25-0.04, 25+0.04, where=z>0,color='k')
ax1.fill_betweenx(z, 42-0.04, 42+0.04, where=z>0,color='k')
ax1.scatter(0,sourceDepth,c='k')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('z [m]')

ax2 = pl.subplot2grid((3,2),(1,0))
ax2.plot(n_profile, z_profile)
ax2.set_xlim(1,2)
ax2.scatter(1,sourceDepth,c='k')
ax2.set_xlabel('n')
ax2.set_ylabel('z [m]')
ax2.set_ylim(15,-2)
ax2.grid()

ax3 = pl.subplot2grid((3,2),(1,1))
ax3.plot(20*np.log10(abs(sim.get_field(x0=42))), sim.get_z())
ax3.set_xlabel('P [dBu]')
ax3.set_ylabel('z [m]')
ax3.set_ylim(15,-2)
ax3.grid()

ax4 = pl.subplot2grid((3,2),(2,0),colspan=2)
ax4.plot(sim.get_x(), 20*np.log10(sim.get_field(z0=sourceDepth)))
ax4.set_xlabel('x [m]')
ax4.set_ylabel('P [dBu]')
ax4.grid()
pl.tight_layout()
pl.show()