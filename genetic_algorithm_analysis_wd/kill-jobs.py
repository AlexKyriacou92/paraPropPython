import os
import subprocess

list_job_cmd = 'squeue | grep "kyriacou" | cat'
#list_job_cmd = 'squeue | cat'

#output = subprocess.check_output('squeue | grep "kyriacou" | cat')
output = subprocess.check_output(list_job_cmd, shell=True)
#print(output)

#
#squeue | grep "kyriacou" | cat
ii = 0
job_list = []
for line in output.split():
    #print(ii, str(line))
    #print(ii%8)
    #ii += 1
    if (ii%8) == 0:
        job_list.append(str(line)[2:-1])
    ii += 1

for i in range(len(job_list)):
    #print(i, int(job_list[i]))
    kill_cmd = 'scancel ' + job_list[i]
    os.system(kill_cmd)
