# paraPropPython

<<<<<<< HEAD
Authors:
Alex Kyriacou - kyriacou@uni-wuppertal.de
Steven Prohira
Cade Sbrocco
=======
Authors: Steven Prohira, Cade Sbrocco, Alex Kyriacou
contact: kyriacou@uni-wuppertal.de
>>>>>>> 1a05c735bb70c7be1f8ebb94a95872b06de87e91

this is a simple parabolic equation EM solver, it uses the parabolic equation approximation to simulate the propagation of EM waves in a medium with a changing index of refraction. 
Currently it is designed to simulate propagation of waves beneath the ice and in the air in the UHF frequency range on relatively short baselines

<<<<<<< HEAD
This is a modified version of the original (arXiv:2011.05997). This version includes a number of updates including:

=======
This is a modified version of the original by Prohira and Cade Sbrocco. The latest version, written by Alex Kyriacou includes a number of updates including:
>>>>>>> 1a05c735bb70c7be1f8ebb94a95872b06de87e91
1. Two dimensional refractive index profiles (range varying)
2. Complex refractive index profiles (models wave attenuation)
3. Backwards reflected waves
4. Adds multi-core processing -> to allow more efficient computation

## Installation

no installation, just clone the repo

### Required packages
The code requires several python modules to run successfully:

These include:
* numpy
* scipy
* matplotlib
* shapely

The recommendation is to create a Anaconda or Miniconda environment. You can download and install Miniconda via: https://docs.conda.io/en/latest/miniconda.html

## using paraPropSimple

cd into the repo directory and try:

python3 simpleExample.py <frequency [GHz, keep it below 1]> <source depth [m]> <use density fluctiations? [0=no, 1=yes]>

and you should see a plot.


## using paraPropPython

Run a Bscan:
python runSim_bscan_from_data.py <fname_config.txt> <fname_n_profile.txt> <fname_output.h5>
