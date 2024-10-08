#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:10:00
#PJM -o result/%j.out
#PJM -e result/%j.err


# ======== Module, Virtualenv and Other Dependencies ======
source ../envs/genkai_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH

export WANDB_API_KEY="412e8240b034b61dae066dfd1b3714cdde7e535e"



# ======== Configuration ========
PROGRAM="src/finetune_wandb.py"
pushd /home/pj24001778/ku40001243/workspace/tangent_task_arithmetic/

# Model & Dataset & Optimizer Setting



count=0
lower_bound=0
upper_bound=100


# Define Setting

PYTHON_ARGS="--port=12370 \
            --finetuning-mode=linear \
            --model=ViT-B-32 \
            --world-size=4 \
            "
# ======== Execution ========
CMD="python ${PROGRAM} ${PYTHON_ARGS}"
echo $CMD
eval $CMD
# --factor_to_schedule='eps'

popd