#!/bin/bash
#SBATCH --job-name=fuse_train
#SBATCH --output=train_fuse_%j.out
#SBATCH --error=train_fuse_%j.err
#SBATCH --time=02:00:00        # Specify run time 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

next_output_dir=$(date +%Y%m%d%H%M%S)
mkdir -p ~/scratch/fuse_train_logs/${next_output_dir}
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
# mkdir -p data/10m/fusion
mkdir -p data/20m
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
# tar -xf $project/data/10m/fusion.tar -C ./data/10m/fusion
tar -xf $project/M3F-Net/data/20m/fusion.tar -C ./data/20m
ls ./data/20m/fusion
echo "Data transfered"

# Load python module, and additional required modules
echo "loading modules..."
module purge
module load python StdEnv gcc arrow

echo "create virtual env..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
# pip install --no-index ray[all]
pip install --no-index ray[tune] tensorboardX lightning pytorch_lightning torch torch-scatter torchaudio torchdata torcheval torchmetrics torchtext torchvision rasterio imageio wandb numpy pandas
pip install seaborn scikit-learn torchsummary geopandas --no-index
pip install pointnext==0.0.5 mamba-ssm[causal-conv1d]==2.2.2
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
srun python train_fuse.py --max_epochs 150 --batch_size 32 --encoder 'xl' --weighted_loss False
#wandb sync ./logs/ray_results/wandb/*

tar -cf ~/scratch/fuse_train_logs/${next_output_dir}/logs.tar ./fuse_train_logs/*
tar -xf ~/scratch/fuse_train_logs/${next_output_dir}/logs.tar -C ~/scratch/fuse_train_logs/${next_output_dir}
mv ~/scratch/fuse_train_logs/${next_output_dir}/fuse_train_logs/ -r ~/scratch/fuse_train_logs/${next_output_dir}

# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    # rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"



