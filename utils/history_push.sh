echo -e "\nGetting a session...\n"
srun -c2 --mem="2GB" -t1:00:00 --pty /bin/bash  -c "
source ~/.bashrc;
echo -e '\nGoogle Drive Sync';
rclone sync '/home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/history/Zips' 'nyu:Data Science/MSERS-Histories' -P;

echo -e '\nOneDrive Sync';
rclone sync '/home/as14229/NYU_HPC/Multilingual-Speech-Emotion-Recognition-System/history/Zips' 'OneDrive:MSERS-Histories' -P;
"
echo -e "\nDone\n"