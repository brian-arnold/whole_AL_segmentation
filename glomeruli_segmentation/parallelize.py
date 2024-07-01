import sys
import gzip
import re
import os
from subprocess import Popen,PIPE
import time

sys.path.insert(0, '../')
from experiment_info import samples, data_dir, puffs, params


def create_job_script(name, outDir, tasks, cpuPerTask, time, mem, command):
	outFile = open('job_%s.sh' % name , 'w')
	o = outDir + "/out." + str(name)
	e = outDir + "/err." + str(name)
	print("#!/bin/bash", file=outFile)
	print("#SBATCH -J "+ str(name), file=outFile)
	print("#SBATCH -o " + o, file=outFile)
	print("#SBATCH -e " + e, file=outFile)
	print("#SBATCH --ntasks=" + str(tasks), file=outFile)
	print("#SBATCH --cpus-per-task=" + str(cpuPerTask), file=outFile)
	print("#SBATCH -t " + str(time), file=outFile)
	print("#SBATCH --mem=" + str(mem), file=outFile)
	print(command, file=outFile)
	outFile.close()
	jobId = sbatch_submit(outFile.name)
	print(jobId)
	os.system("mv job_" + str(name) + ".sh " + outDir)
	return(jobId)

#Submit filename to slurm with sbatch, returns job id number
def sbatch_submit(filename):
	proc = Popen('sbatch %s'%filename,shell=True,stdout=PIPE,stderr=PIPE)
	stdout,stderr = proc.communicate()
	stdout = stdout.decode("utf-8","ignore")
	stdout = stdout.strip()
	stdout = stdout.strip('Submitted batch job ')
	return(stdout)



def main():

	conda_environment = "caiman"
	tasks = 1  # total number of tasks across all nodes
	cpu_per_task = 20 # cpu-cores per task (>1 if multi-threaded tasks)
	t = "0-01:00:00" # DAY-HR:MIN"SEC
	mem = 80000 # in MB
	out_dir = "./"

	for i,samp in enumerate(samples):
		if i <= 4:
			continue
		command = "source /mnt/cup/labs/mcbride/bjarnold/miniforge3/etc/profile.d/conda.sh\n" 
		command += f"conda activate {conda_environment}\n" 

		# command += f"python 00_caiman_segment.py {i} 5 4 4 1\n"
		command += f"python 00_caiman_segment.py "
		command += f"--sample_index {i} "
		command += f"--K 8 "
		command += f"--gSig 4 4 1 "
		command += f"--odor_file odor_subset_lists/5_aldehydes_with_refs.txt "
		command += f"--out_dir results/caiman/odor_subset_5_aldehydes "

		print(command)
		create_job_script(f"{i}_8_441", out_dir, tasks, cpu_per_task, t, mem, command)
		time.sleep(1)

if __name__ == '__main__':
  main()