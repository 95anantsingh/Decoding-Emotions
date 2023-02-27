#!/bin/bash
module purge
source ~/.bashrc
conda activate NLP_Nightly
cd /home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python extracter.py