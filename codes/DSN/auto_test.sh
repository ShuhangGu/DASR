#/usr/bin/zsh

cd /media/4T/Dizzy/real-world-sr/dsgan

python create_dataset_modified.py --checkpoint /media/4T/Dizzy/real-world-sr/dsgan/checkpoints/0621_Deresnet+Gau+FSD+tex0.01/last_iteration.tar --Generator DeResnet --Discriminator FSD --dataset aim2019 --name 0622_AIM_FSD_Gau_wtex0.01

python create_dataset_modified.py --checkpoint /media/4T/Dizzy/real-world-sr/dsgan/checkpoints/0621_Deresnet+Gau+FSD+tex0.02/last_iteration.tar --Generator DeResnet --Discriminator FSD --dataset aim2019 --name 0622_AIM_FSD_Gau_wtex0.02

python create_dataset_modified.py --checkpoint /media/4T/Dizzy/real-world-sr/dsgan/checkpoints/0621_Deresnet+Gau+FSD+tex0.04/last_iteration.tar --Generator DeResnet --Discriminator FSD --dataset aim2019 --name 0622_AIM_FSD_Gau_wtex0.04

python create_dataset_modified.py --checkpoint /media/4T/Dizzy/real-world-sr/dsgan/checkpoints/0621_Deresnet+Gau+FSD+tex0.009/last_iteration.tar --Generator DeResnet --Discriminator FSD --dataset aim2019 --name 0622_AIM_FSD_Gau_wtex0.009