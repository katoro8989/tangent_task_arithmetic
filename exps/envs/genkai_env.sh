# Pyenv VirtualEnv Environment
PYTHON_PATH="/home/pj24001778/ku40001243/env/py3916/bin"
export PYTHON_PATH

# Cluster
CLUSTER_NAME="genkai_cluster"
export CLUSTER_NAME

# # ======== Modules ========

# T4

module load cuda/12.1.0
module load cudnn/9.0.0
module load nccl/2.20.5

eval "module list"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/11.8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/11.8.0/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/apps/t4/rhel9/free/cudnn/9.0.0/cuda/12/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/t4/rhel9/free/cudnn/8.9.7/cuda/11/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/t4/rhel9/free/nccl/2.20.5/cuda12.3.2/lib:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH

eval "nvidia-smi"
