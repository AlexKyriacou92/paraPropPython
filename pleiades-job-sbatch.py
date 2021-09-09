import os
import sys

"""
Information:
This script creates a list a bash scripts to run parabolic equation simulation jobs on a slurm cluster

Input:
A list of Time Domain or Mono-Frequency PE simulations with a predefined refractive index profile, geometry and pulse, where each job is on a different line, and these correspond to difference source positions

Output:
For each line, a bash script (i.e. src=10.sh) is created, which contains the allocated nodes, partition, allowed time and memory allocated to each job
"""

#Input Arguments
fname_in = sys.argv[1] #List of jobs to be executed
path2jobs = sys.argv[2] #Directory to save bash scripts to

#Cluster Settings
NODES_MIN = 4
NODES_MAX = 100
PARTITION = 'long'
DAYS = 3
HOURS = 0
MEMORY = 400 # in MB

def make_sbatch(jobline, fname, jobname, nNodes_min, nNodes_max, partition, days, hours, nodeMemory): #Make batch file to execute job
    sbatch = "#SBATCH"
    fout = open(fname, 'w+')
    fout.write("#!/bin/sh\n")

    minutes = 0
    seconds = 0

    fout.write(sbatch + " --job-name=" + jobname +"\n")
    fout.write(sbatch + " --partition=" + partition + "\n")
    fout.write(sbatch + " --time=" +str(days) + "-" + str(hours) + ":" + str(minutes) + ":" + str(seconds) + " # days-hours:minutes:seconds\n")
    if nNodes_min == nNodes_max:
        fout.write(sbatch + " --nodes=" + str(nNodes_min) + "\n")
    else:
        fout.write(sbatch + " --nodes=" + str(nNodes_min) + "-" + str(nNodes_max) + "\n")
    fout.write(sbatch + " --mem-per-cpu=" + str(nodeMemory) + " # in MB\n")
    fout.write(jobline)

    makeprogram = "chmod u+x " + fname
    os.system(makeprogram)

if __name__ == "__main__":
    fin = open(fname_in, "r+")
    for jobline in fin:
        cols = jobline.split()
        src_depth = cols[3]
        jobname = "src" + src_depth
        sbatch_file = path2jobs + "/src" + src_depth + ".sh"
        make_sbatch(jobline, sbatch_file, jobname, NODES_MIN, NODES_MAX, PARTITION, DAYS, HOURS, MEMORY)
    fin.close()