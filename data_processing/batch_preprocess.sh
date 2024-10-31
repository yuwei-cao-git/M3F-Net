#!/bin/bash
#SBATCH --job-name=preprocess_data  # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=4                  # Number of tasks (one per GPU)
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --gres=gpu:4                # Number of GPUs
#SBATCH --time=02:00:00             # Time limit
#SBATCH --output=preprocess_%j.out  # Standard output log
#SBATCH --error=preprocess_%j.err   # Standard error log

module load anaconda3/2022.05
module load cuda/11.3

# Paths to the season folders
seasons=("spring" "summer" "autumn" "winter")

# Run the S2_sen2cor.py script on each season folder
for i in {0..3}; do
   srun --ntasks=1 --gres=gpu:1 python S2_sen2cor.py ${seasons[$i]} &
done

wait
