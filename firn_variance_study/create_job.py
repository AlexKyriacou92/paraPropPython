import sys
import ku_scripting
import os
import configparser

from ku_scripting import make_job

cluster_config = sys.argv[1]

config = configparser.ConfigParser()
config.read(cluster_config)

#Cluster Settings
cluster_settings = config['CLUSTER']

NODES_MIN = int(cluster_settings['NODES_MIN'])
NODES_MAX = int(cluster_settings['NODES_MAX'])

PARTITION = cluster_settings['PARTITION']
DAYS = int(cluster_settings['DAYS'])
HOURS = int(cluster_settings['HOURS'])
MEMORY = int(cluster_settings['MEMORY']) # in MB
TASKS = int(cluster_settings['TASKS'])
NCPUS = int(cluster_settings['NCPUS'])
print('TASKS=', TASKS, 'NPCUS=', NCPUS)

job_settings = config['JOB']
operator = job_settings['operator']
script = job_settings['script']
arg = job_settings['arg']
prefix = job_settings['prefix']
mpi = job_settings['mpi']

py_cmd = operator + ' ' + script + ' ' + arg
if mpi == 'mpirun':
    cmd = mpi + ' -n ' + str(TASKS) + ' ' + py_cmd
else:
    cmd = py_cmd

fname_shell = prefix + '.sh'
fname_out = prefix + '.out'
job_name = prefix
make_job(fname_shell=fname_shell, fname_outfile=fname_out, jobname=job_name, command=cmd,
         nNodes_min=NODES_MIN, nNodes_max=NODES_MAX, partition=PARTITION,
         days=DAYS, hours=HOURS, nodeMemory=MEMORY, tasks = TASKS, ncpus=NCPUS)