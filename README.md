# DASR
(CVPR-2021) Official PyTorch code for our  paper DASR: [Unsupervised Real-world Image Super Resolution via 
Domain-distance Aware Training](https://arxiv.org/abs/2004.01178). 


### Abstract
These days, unsupervised super-resolution (SR) has been soaring due to its practical and promising potential in real scenarios. The philosophy of off-the-shelf approaches lies in the augmentation of unpaired data, \ie  first generating synthetic low-resolution (LR) images $\mathcal{Y}^g$ corresponding to real-world high-resolution (HR) images $\mathcal{X}^r$ in the real-world LR domain $\mathcal{Y}^r$, and then utilizing the pseudo pairs $\{\mathcal{Y}^g, \mathcal{X}^r\}$ for training in a supervised manner. Unfortunately, since image translation itself is an extremely challenging task, the SR performance of these approaches are severely  limited by the domain gap between generated synthetic LR images and real LR images. In this paper, we propose a novel domain-distance aware super-resolution (DASR) approach for unsupervised real-world image SR. The domain gap between training data (e.g. $\mathcal{Y}^g$) and testing data (e.g. $\mathcal{Y}^r$) is addressed with our \textbf{domain-gap aware training} and \textbf{domain-distance weighted supervision} strategies. Domain-gap aware training takes additional benefit from real data in the target domain while domain-distance weighted supervision brings forward the more rational use of labeled source domain data. The proposed method is validated on synthetic and real datasets and the experimental results show that DASR consistently outperforms state-of-the-art unsupervised SR approaches in generating SR outputs with  more realistic and natural textures. Code will be available at [DASR](https://github.com/ShuhangGu/DASR).

# Requirements

- Pytorch == 1.1.0
- torchvision == 0.3.0
- opencv-python
- tensorboardX

# Usage
### Hierarchy of DASR codes
```
DASR:
|
|----codes
|       |---- DSN
|       |---- SRN
|       |---- PerceptualSimilarity
|
|----DSN_experiments
|----SRN_experiments
|----DSN_tb_logger
|----SRN_tb_logger
|----DSN_results
|----SRN_results
```



### Data Preparation
Please specify the path to your dataset in `paths.yml`. 

We followed the AIM2019 challenge descriptions that `source domain` indicates the 
noisy and low-resolution and `target domain` indicates the clean 
and high resolution  images. The images in source domain and target domain
are ***unpaired***.

### Auto Reproducing
Auto reproduce process can be presented as  
`Training DSN   -->
Generating LRs and domain distance maps   -->  Training SRN`.

Our results can be reproduced by running a single command:
```
cd DASR/codes
python Auto_Reproduce.py --dataset aim2019/realsr  \
                         --artifact tdsr/(tddiv2k/tdrealsr)
```
Auto-reproducing process will take about 48 hours on Nvidia GTX 1080.

### Pretrained model
We provide pretrained models for AIM2019/ RealSR and CameraSR datasets.


| |DSN | SRN|
|---|:---:|:---:|
|AIM|[DeResnet](https://drive.google.com/file/d/1egzDbeL3UXeDwrjIapIL2HMtHEDImLSr/view?usp=sharing)|[ESRGAN](https://drive.google.com/file/d/1vOcFD1nfm9AVcU5xzFCnphMydCBPtYfZ/view?usp=sharing)|
|RealSR|[DeResnet](https://drive.google.com/file/d/1tbAgx0r50Y8aUrxbfcqCtT1sfSor8Jff/view?usp=sharing)|[ESRGAN](https://drive.google.com/file/d/1N9bGLoHrOl3WnWG5eQBWFdvvbLLP5zQM/view?usp=sharing)|
|CameraSR|[DeResnet](https://drive.google.com/file/d/1IqcKqOJ2ZZrCbGV2CJbbX2QR86TvJ2dB/view?usp=sharing)|[ESRGAN](https://drive.google.com/file/d/1gk5YsbY_976ZU69eVq-bVkMbM2-LE5A8/view?usp=sharing)|


### Testing 

Testing is similar to the [BasicSR](https://github.com/xinntao/BasicSR) codes.
We add some extra options including `LPIPS`,`Forward_chop`, etc.
Please specify the pre-trained model and data path in
`test_sr.json` correctly.
```
cd DASR/codes/SRN
python test.py -opt options/test/test_sr.json
```


### DownSampling Network (DSN)
We build our DSN codes based on our previous work 
FSSR ([github](https://github.com/ManuelFritsche/real-world-sr/tree/master/dsgan)
| [paper](https://arxiv.org/abs/1911.07850)).

#### Training
The DSN model can be trained with:
```
cd DASR/codes/DSN
python train.py --dataset aim2019 --artifacts tdsr \
                --generator DeResnet --discriminator FSD \
                --norm_layer Instance \
                --filter wavelet --cat_or_sum cat  \
                --batch_size 8 --num_workers 8 --crop_size 256 \
                --save_path 0603_DeResnet+wavcat+FSD+wtex0.03
```
Args:

- `--dataset` The dataset should be one of the `aim2019`/`realsr`/`camerasr`.
To train DSN on your own data, please specify the path
in the correct format in `paths.yml`.

- `--aritifact` Choose the type of degradation artifact that you
specified in `paths.yml`.

- `--generator` Choose the downsampling generator architecture, including
`DeResnet`, `DSGAN`.

- `--discriminator` Choose the discriminator architecture including
`nld_s1`(Nlayer discriminator with stride 1), `nld_s2` and `FSD` (Frequency separation 
discriminator).

- `--norm_layer` Choose the normalization layer in discriminator, including 
`Instance` and `Batch`.

- `--filter` Choose the frequency separation filter, including 
`wavelet`, `gaussion` and `avg_pool`.

- `--cat_or_sum` Choose the approach of combination of wavelet subbands,
including  `cat` and `sum`.

The training loss and validation results are shown in `DASR/DSN_tb_logger`
by running with:
```
cd DASR/DSN_tb_logger
tensorboard --logdir=./
```


#### Generate LR-HR pairs and domain-distance maps.

The training data for SRN can be generated with:
```
python create_dataset_modified.py --dataset aim2019 \
                                  --checkpoint <path to your DSN model> \
                                  --generator DeResnet --discriminator FSD  --filter avg_pool\
                                  --name 0603_DSN_LRs_aim2019
```
Please note that our generator and discriminator are saved in a single
`.tar` files.

The generated low-resolution images and domain distance maps are 
saved in `DASR/DSN_results/<--name>/`.


### Super-Resolution Network (SRN)
We build our SRN codes based on [BasicSR](https://github.com/xinntao/BasicSR) 
super-resolution toolkit.

#### Training
The SRN model can be trained with:
```
cd DASR
python codes/train.py -opt codes/options/train/train_DASR.json
```
Please specify the configurations in `train_DASR.json` file.



### Citation
```
@article{wei2020unsupervised,
  title={Unsupervised Real-world Image Super Resolution via Domain-distance Aware Training},
  author={Wei, Yunxuan and Gu, Shuhang and Li, Yawei and Jin, Longcun},
  journal={arXiv preprint arXiv:2004.01178},
  year={2020}
}
```