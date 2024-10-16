#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --gpus-per-node=4            # number of gpus per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --tasks-per-node=4 # This is the number of model replicas we will place on the GPU.
#SBATCH --mem=128G
#SBATCH --job-name="test-multi-gpu"
#SBATCH --time=03:00:00        # Specify run time 
#SBATCH --mail-user=yuwei.cao@ubc.ca    # Request email notifications
#SBATCH --mail-type=ALL

next_output_dir=$(date +%Y%m%d%H%M%S)
mkdir ~/scratch/output/${next_output_dir}
echo "created output dir"

#SBATCH --output=~/scratch/output/${next_output_dir}/%N-%j.out    # Specify output file format generated by python script
#SBATCH --error=~/scratch/output/${next_output_dir}/%N-%j.error

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
mkdir -p data/10m
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/data/10m.tar -C ./data/10m
echo "Data transfered"

# Load python module, and additional required modules
module purge 
module load python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
#pip install lightning torcheval --no-index
pip install --no-index -r requirements.txt
pip install laspy[laszip]
echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
wandb login *

#Run python script
# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
echo "Start runing model.................................................................................."
srun python train.py --data_dir './data' --resolution 10 --log_name 'ResUnet_S4_10m_MF' --num_epoch 200 --batch_size 48 --mode 'img' --use_mf --use_residual

cd $SLURM_TMPDIR
tar -cf ~/scratch/output/${next_output_dir}/checkpoints.tar ./logs/*
tar -cf ~/scratch/output/${next_output_dir}/wandblogs.tar ./wandb/*

# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"