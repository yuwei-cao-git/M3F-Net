common commands:
```
ssh -Y ycao68@cedar.alliancecan.ca

cd $project/M3F-Net
git pull

module purge
module load python StdEnv gcc arrow

$ salloc --time=2:0:0 --gpus=2 --mem-per-gpu=32G --ntasks=1
# salloc --gres=gpu:1 --cpus-per-task=8 --mem=32000M --time=1:00:00
source ~/env/bin/activate
# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

srun python ray_tune.py
srun python train_pts.py

```
1. torch_scatter
File "/mnt/d/Sync/research/tree_species_estimation/code/fusion/M3F_Net/data_utils/common.py", line 5, in <module>
    from torch_scatter import scatter_mean
  File "/home/yuwei-linux/code/venv/lib/python3.10/site-packages/torch_scatter/__init__.py", line 16, in <module>
    torch.ops.load_library(spec.origin)
  File "/home/yuwei-linux/code/venv/lib/python3.10/site-packages/torch/_ops.py", line 1295, in load_library
    ctypes.CDLL(path)
  File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /home/yuwei-linux/code/venv/lib/python3.10/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev

fix: pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html --no-cache-dir