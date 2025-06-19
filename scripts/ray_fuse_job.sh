#!/bin/bash
#SBATCH --job-name=ray_fuse_tune
#SBATCH --output=ray_fuse_tune_%j.out
#SBATCH --time=1-12:00:00        # Specify run time 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

mkdir -p ~/scratch/fuse_tune_logs
echo "created output dir"

# Trap the exit status of the job
trap 'job_failed=$?' EXIT

# code transfer
cd $SLURM_TMPDIR
mkdir -p work/M3F-Net
cd work/M3F-Net
cp -r $project/SRS-Net/dataset .
cp -r $project/SRS-Net/data_utils .
cp -r $project/SRS-Net/models .
cp -r $project/SRS-Net/utils .
cp -r $project/SRS-Net/tune_fuse_ovf.py .
echo "Source code cloned!"

echo "Start transfer data..."
# mkdir -p data/10m/fusion
mkdir -p data/20m
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
# tar -xf $project/data/10m/fusion.tar -C ./data/10m/fusion
tar -xf $project/M3F-Net/data/20m/fusion_v2.tar -C ./data/20m
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
pip install --no-index ray[tune] lightning pytorch_lightning torch==2.5.0 tensorboardX torch-scatter torchaudio torchdata torcheval torchmetrics torchtext torchvision rasterio imageio wandb numpy pandas
pip install --no-index seaborn scikit-learn torchsummary geopandas
pip install --no-index pointnext mamba-ssm
pip install --no-index laspy[laszip]

echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
wandb offline

#Run python script
echo "Start runing model.................................................................................."
srun python tune_fuse_ovf.py --max_epochs 200
#wandb sync ./logs/ray_results/wandb/*

tar -cf ~/scratch/fuse_tune_logs/logs.tar ./fuse_tune_logs/ray_results/*
tar -xf ~/scratch/fuse_tune_logs/logs.tar -C ~/scratch/fuse_tune_logs
rm ~/scratch/fuse_tune_logs/logs.tar
ls ~/scratch/fuse_tune_logs/
mv ~/scratch/fuse_tune_logs/logs/ ~/scratch/fuse_tune_logs/


# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"
