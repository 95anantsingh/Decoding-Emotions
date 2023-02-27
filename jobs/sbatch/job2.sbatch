#!/bin/bash

#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4       
#SBATCH --time=3:00:00          
#SBATCH --mem=128GB              
#SBATCH --gres=gpu:1       
#SBATCH --job-name=Job2
#SBATCH --output=/home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/jobs/logs/batch_2/job2.log

module purge
source ~/.bashrc
conda activate NLP_Nightly
cd /home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/

python main.py -ll debug -em disk -d AESDD -m WAV2VEC2_BASE -r Run2
