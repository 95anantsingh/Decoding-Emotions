#!/bin/bash

#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=gpu:1            
#SBATCH --partition=a100_1,a100_2,rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --time=00:30:00
#SBATCH --job-name=Job152
#SBATCH --output=/home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/jobs/logs/batch_1/job152.log

source ~/.bashrc
conda activate MSERS
cd /home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/

python main.py -ll debug -em gpu_memory -nw 3 -cm DENSE -e 30  -d RAVDESS -fm HUBERT_ASR_LARGE -r Run1 -jn Job152