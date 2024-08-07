#!/bin/bash
#SBATCH -J run
#SBATCH -o out
#SBATCH -e err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G         
#SBATCH --time 1-00:00:00        # DAYS-HOURS:MINUTES:SECONDS

source /jukebox/mcbride/bjarnold/miniforge3/etc/profile.d/conda.sh

conda activate caiman
python 01_segment_and_extract_traces.py
# python caiman_test.py
