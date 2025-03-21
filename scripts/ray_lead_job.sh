#!/bin/bash
#SBATCH --job-name=ray_fuse_tune
#SBATCH --output=ray_lead_tune_%j.out
#SBATCH --error=ray_lead_tune_%j.err
#SBATCH --time=20:00:00        # Specify run time 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

mkdir -p ~/scratch/lead_tune_logs
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

echo "Start transfer data..."
mkdir -p data/20m
tar -xf $project/M3F-Net/data/20m/fusion.tar -C ./data/20m
ls ./data/20m/fusion_v2
# ls ./data/20m/fusion/train/superpixel
echo "Data transfered"

# Load python module, and additional required modules
echo "loading modules..."
module --force purge
module load python StdEnv gcc arrow cuda

echo "create virtual env..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index ray[tune] tensorboardX lightning pytorch_lightning torch torch-scatter torchaudio torchdata torcheval torchmetrics torchtext torchvision rasterio imageio wandb numpy pandas
pip install seaborn scikit-learn torchsummary geopandas --no-index
pip install pointnext==0.0.5 mamba-ssm==2.2.2
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
srun python tune_leading_classify.py --max_epochs 200 --n_samples 30 --task "classify"
#wandb sync ./logs/ray_results/wandb/*

tar -cf ~/scratch/lead_tune_logs/logs.tar ./lead_tune_logs/ray_results/*
tar -xf ~/scratch/lead_tune_logs/logs.tar -C ~/scratch/lead_tune_logs
rm ~/scratch/lead_tune_logs/logs.tar
ls ~/scratch/lead_tune_logs/
mv ~/scratch/lead_tune_logs/logs/ ~/scratch/lead_tune_logs/


# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"
