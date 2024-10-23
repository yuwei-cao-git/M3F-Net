common commands:
```
ssh -Y ycao68@cedar.alliancecan.ca
git pull
cd Pytorch/models/PointAugment
module purge
module load python StdEnv gcc arrow
source ~/env/bin/activate
$ salloc --time=1:0:0 --gpus=2 --mem-per-gpu=32G --ntasks=2
# salloc --gres=gpu:1 --cpus-per-task=8 --mem=32000M --time=1:00:00
# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
```
