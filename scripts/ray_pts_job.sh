#!/bin/bash
#SBATCH --job-name=ray_pts_tune_pts
#SBATCH --output=ray_pts_tune_%j.out
#SBATCH --error=ray_tune_%j.err
#SBATCH --time=05:30:00        # Specify run time 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

next_output_dir=$(date +%Y%m%d%H%M%S)
mkdir -p ~/scratch/pts_tune_log/${next_output_dir}
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
mkdir -p data/rmf_laz
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/M3F-Net/data/rmf_laz.tar -C ./data
echo "Data transfered"

module purge
# module load gcc/9.3.0 arrow python/3.10 scipy-stack/2022a
module load python StdEnv gcc arrow

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip                                                                        
#pip install --no-index ray[all]
pip install --no-index ray[tune] tensorboardX lightning pytorch_lightning torch torch-scatter torchaudio torchdata torcheval torchmetrics torchtext torchvision rasterio imageio wandb numpy pandas
pip install seaborn scikit-learn torchsummary geopandas --no-index
pip install pointnext==0.0.5
pip install laspy[laszip]

echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
export WANDB_API_KEY=*
wandb login

#Run python script
echo "Start runing model.................................................................................."
srun python tune_pts.py --max_epochs 200
#wandb sync ./logs/ray_results/wandb/*

tar -cf ~/scratch/pts_tune_log/${next_output_dir}/tmp.tar /tmp/ray/*
tar -cf ~/scratch/pts_tune_log/${next_output_dir}/logs.tar ./pts_tune_logs/ray_results/*

# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"