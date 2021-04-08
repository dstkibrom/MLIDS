#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
export PATH="/work/araya-kd/anaconda3/bin:$PATH"
source activate tf_gpu_cuda8
cd /work/araya-kd/MLIDS/train_arbids
python3 train_arbid_34.py
