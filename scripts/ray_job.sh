#!/bin/bash
#SBATCH --job-name=ray_tune
#SBATCH --output=ray_tune_%j.out
#SBATCH --error=ray_tune_%j.err
#SBATCH --time=03:00:00        # Specify run time 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

next_output_dir=$(date +%Y%m%d%H%M%S)
mkdir ~/scratch/output/${next_output_dir}
echo "created output dir"

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
mkdir -p data/20m
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/data/20m.tar -C ./data/20m
echo "Data transfered"

# Load python module, and additional required modules
module purge 
module load python/3.11 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install ray numpy torch torchaudio pytorch_lightning lightning torcheval --no-index
pip install --no-index -r requirements.txt
pip install laspy[laszip]

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
wandb login *
#wandb agent ubc-yuwei-cao/M3F-Net/kfqtbh8r

#Run python script
# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
echo "Start runing model.................................................................................."
srun python train.py

cd $SLURM_TMPDIR
tar -cf ~/scratch/output/${next_output_dir}/checkpoints.tar ./logs/checkpoints/*
tar -cf ~/scratch/output/${next_output_dir}/wandblogs.tar ./wandb/*

# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"