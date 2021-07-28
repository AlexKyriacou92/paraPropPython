# paraPropPython

this is a simple parabolic equation EM solver. It uses the parabolic equation approximation to simulate the propagation of EM waves in a medium with a changing index of refraction. 

Currently it is designed to simulate propagation of waves beneath the ice and in the air on short-ish baselines. 

2 n(z) profiles can be used: south pole, and taylor dome antarctica. These functional forms can be used as smooth n(z) profiles, or can be augmented with experimental data (for the south pole) and random density fluctuations (for the taylor dome site, data hopefully forthcoming) to simulate realistic RF propagation in the firn. 

It is written in python (which i don't really speak), is new, and is being expanded, and will probably break a lot. email prohira.1 attt osu dottt edu with questions.

## installing

no installation, just clone the repo

## using

cd into the repo directory and try:

python3 simpleExample.py <frequency [GHz, keep it below 1]> <source depth [m]> <use density fluctiations? [0=no, 1=yes]>

and you should see a plot. 

# paraPropBackwards
this is an update of the original code to include a range dependent refractive index (n = n(x,z)) and to allow simulation of backwards scattered EM waves off of objects by implementing a 2 way split step parabolic equation equation solver.

Simulation data, metadata and settings are haved to HDF files
Simulations are set up using a 'input' text file. See example.
To run a simulation:
python runSimulation.py <input-file.txt> <output-file.h5>
