#!/bin/bash

#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=gpu:1            
#SBATCH --time=2:45:00
#SBATCH --partition=a100_1,a100_2,rtx8000,v100
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --job-name=Job36
#SBATCH --output=/home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/jobs/logs/batch_1/job36.log

module purge
source ~/.bashrc
conda activate NLP_Nightly
cd /home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/

python main.py -ll debug -em gpu_memory -nw 2 -cm DENSE  -d IEMOCAP -m WAV2VEC2_LARGE_XLSR300M -r Run5 -jn Job36