#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
export PATH="/work/araya-kd/anaconda3/bin:$PATH"
source activate tf_gpu
cd /work/araya-kd/MLIDS/train_arbids
python3 train_arbid_11.py
