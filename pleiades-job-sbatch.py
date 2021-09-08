import numpy as np
import os
import sys
import datetime

if len(sys.argv) != 3:
    print("error!, you must supply a list of jobs to be run \n format should look like: \n python pleaides-job.sbathc.py job-list.txt jobname")
    sys.exit()

joblist = str(sys.argv[1])

nNodes_min = int(8)
nNodes_max = int(300)

nodeMemory = int(350) #need to guess this before hand, should be in MB

jobname = str(sys.argv[2])
days = int(3.)
hours = int(0)
minutes = int(0)
seconds = int(0)
partition = 'normal'

sbatch = "#SBATCH"
fname = jobname + '.sh'

fout = open(fname, 'w+')
fout.write("#!bin/sh\n")

fout.write(sbatch + " --job-name=" + jobname +"\n")
fout.write(sbatch + " --partition=" + partition + "\n")
fout.write(sbatch + " --time=" +str(days) + "-" + str(hours) + ":" + str(minutes) + ":" + str(seconds) + " # days-hours:minutes:seconds\n")
fout.write(sbatch + " --nodes=" + str(nNodes_min) + "-" + str(nNodes_max) + "\n")
fout.write(sbatch + " --mem-per-cpu=" + str(nodeMemory) + " # in MB\n")

fin = open(joblist, "r+")
for line in fin:
    fout.write(line)
fin.close()
fout.close()

makeprogram = "chmod u+x " + fname
os.system(makeprogram)