# To save you reproduce time, we use the following settings to training DSN.
# It may cause a slight performance loss but still comparable.

python train.py --dataset aim2019 --artifacts tdsr --generator DeResnet --discriminator FSD --filter avg_pool  \
                --w_tex 0.006 --save_path 0603_DSN_aim2019 \
                --batch_size 8 --num_workers 32 --crop_size 256

python create_dataset_modified.py --dataset aim2019 \
                                  --checkpoint ../../DSN_experiments/0603_DSN_aim2019/checkpoints/last_iteration.tar \
                                  --generator DeResnet --discriminator FSD  --filter avg_pool\
                                  --name 0603_DSN_LRs_aim2019