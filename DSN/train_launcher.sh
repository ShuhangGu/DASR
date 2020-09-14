python train.py --dataset aim2019 --artifacts tdsr \
                --generator DeResnet --discriminator FSD --norm_layer Instance \
                --filter wavelet --cat_or_sum cat  \
                --batch_size 8 --num_workers 8 --crop_size 256 \
                --save_path 0618_DeResnet+wavcat+FSD+wtex0.03

