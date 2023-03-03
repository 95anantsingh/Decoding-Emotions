#!/bin/bash

#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=gpu:1            
#SBATCH --partition=a100_1,a100_2,rtx8000,v100,gpu_misc_v100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:15:00
#SBATCH --job-name=Job1
#SBATCH --output=/home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/jobs/logs/batch_1/job1.log

module purge
source ~/.bashrc
conda activate NLP_Nightly
cd /home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/

python main.py -ll debug -em gpu_memory -nw 2 -cm PROBING -e 20  -d AESDD -fm WAV2VEC2_BASE -r Run1 -jn Job1