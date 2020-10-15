# To save you reproduce time, we use the following settings to training DSN.
# It may cause a slight performance loss but still comparable.

python train.py --dataset realsr --artifacts tdrealsr --generator DeResnet --filter avg_pool  \
                --w_tex 0.005 --save_path 0603_DSN_realsr \
                --batch_size 8 --num_workers 8 --crop_size 128

python create_dataset_modified.py --dataset realsr_tdrealsr \
                                  --checkpoint ../../DSN_experiments/0603_DSN/checkpoints/last_iteration.tar \
                                  --generator DeResnet --discriminator FSD  \
                                  --name 0603_DSN_LRs_realsr