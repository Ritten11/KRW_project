#!/bin/bash

#SBATCH --job-name=RDF2Vec_embedding
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=defq

## in the list above, the partition name depends on where you are running your job. 
## On DAS5 the default would be `defq` on Lisa the default would be `gpu` or `gpu_shared`
## Typing `sinfo` on the server command line gives a column called PARTITION.  There, one can find the name of a specific node, the state (down, alloc, idle etc), the availability and how long is the time limit . Ask your supervisor before running jobs on queues you do not know.

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate KRW_project-3.11.3

# Run the actual experiment. 
python /var/scratch/rro252/KRW_project/create_embedding.py -id $SLURM_ARRAY_TASK_ID -e [25,100,200,500] -kg tax_subset_label_NCIT -w 24
