#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
export PATH="/work/araya-kd/anaconda3/bin:$PATH"
source activate tf_gpu_cuda8
cd /work/araya-kd/MLIDS/MLIDS_offline
python3 arb_id_33.py