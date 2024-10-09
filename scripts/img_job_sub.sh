#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --gpus-per-node=4            # number of gpus per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --tasks-per-node=4 # This is the number of model replicas we will place on the GPU.
#SBATCH --mem=128G
#SBATCH --job-name="test-multi-gpu"
#SBATCH --time=01:00:00        # Specify run time 
#SBATCH --mail-user=yuwei.cao@ubc.ca    # Request email notifications
#SBATCH --mail-type=ALL

# Get the highest existing index
highest_index=$(ls -d output/run*/ 2>/dev/null | grep -o '[0-100]*' | sort -n | tail -1)

# If no output folders exist, start from 1
if [ -z "$highest_index" ]; then
    next_index=1
else
    next_index=$((highest_index + 1))
fi
# Create the next output folder
mkdir "logs/run${next_index}"
echo "Created folder: logs/run${next_index}"
next_output_dir="logs/run${next_index}"

#SBATCH --output=${next_output_dir}/%N-%j.out    # Specify output file format generated by python script
#SBATCH --error=${next_output_dir}%N-%j.error

# Trap the exit status of the job
trap 'job_failed=$?' EXIT

# code transfer
cd $SLURM_TMPDIR
mkdir work
cd work
git clone git@github.com:yuwei-cao-git/M3F-Net.git
cd M3F-Net
echo "Source code cloned!"

# data transfer
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/data/20m.tar -C ./data
echo "Data transfered"

# Load python module, and additional required modules
module purge 
module load python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install laspy[laszip]
echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

#Print (echo) info to output file
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching test dl cc script"
# 3mins so far

# Log experiment variables
wandb login xxx

#Run python script
# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
echo "Start runing model.................................................................................."
srun python main.py --init_method tcp://$MASTER_ADDR:3456 --data_dir 'data' --resolution 20 --log_name 'ResUnet_S4_20m_MF' --num_epoch 200 --batch_size 48 --mode 'img' --use_mf --use_residual

cd $SLURM_TMPDIR
tar -cf ~/scratch/output/run${next_index}/checkpoints.tar work/M3F-Net/logs/checkpoints/*
tar -cf ~/scratch/output/run${next_index}/wandblogs.tar work/M3F0Bet/logs/wandb/*

# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"