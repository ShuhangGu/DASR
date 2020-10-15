# DSGAN
A PyTorch implementation of the DSGAN model, which allows to create
datasets for Super-Resolution. The model is trained in an unsupervised
fashion on the HR images. It then learns to generate the appropriate 
LR images, which have the same corruptions as the original HR images.

## Requirements
- install [Pytorch](https://pytorch.org/) according to their documentation
- install other dependencies:
```
conda install -y numpy pyyaml tqdm pillow
pip install tensorboardX
```
- download [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and 
save it at the same level as this repository. Alternatively, change the path
in line 7 of `loss.py`.
- For visualisation use tensorboard

## Usage
### Train
Add the path to your dataset to `paths.yml` as follows:
```yaml
dataset:
  artifacts:
    hr:
      train: 'path/to/HR/train/data'
      valid: 'path/to/HR/validation/data'
    lr:
      train: 'OPTIONAL/path/to/LR/train/data'
      valid: 'OPTIONAL/path/to/LR/validation/data'    
```
The model can then be trained with:
```
python train.py --dataset dataset --artifacts artifacts
```
Thereby, `dataset` and `artifacts` correspond to the first and second line 
of the `paths.yml` file example, respectively. 
Use `python train.py -h` to view optional arguments.

### Create dataset
You can use the trained parameters (saved in the folder `checkpoints`) to
create a dataset from your original images. To do so, you need to first add
your desired dataset location to `paths.yml` as follows:
```yaml
datasets:
  dataset: 'path/to/desired/location'
```
The dataset can then be created with:
```
python create_dataset.py --dataset dataset --checkpoint checkpoint
```
Two datasets will be generated, one for the SDSR case and one for the TDSR case.
Use `python create_dataset.py -h` to view optional arguments.

### Pre-Trained Models
We provide the pre-trained models only for the SDSR case. However, for the TDSR case we used the same models to introduce the artificial corruptions.
| |[DSGAN](https://github.com/ManuelFritsche/real-world-sr/tree/master/dsgan)|
|---|:---:|
|DF2K Gaussian|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/DF2K_gaussian.tar)|
|DF2K JPEG|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/DF2K_jpeg.tar)|
|DPED|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/DPED.tar)|
|AIM 2019|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/AIM2019.tar)|

## Setup for Reproducing the Results
### Datasets 

#### DF2K
- Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset (bicubic x4)
- Download the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset (bicubic x4)
- Combine the HR and LR images of DIV2K and Flickr2K to create DF2K
- Adapt the paths in `paths.yml` if necessary
- Add Artificial corruptions as described below

#### DPED
 - Download the [DPED](http://people.ee.ethz.ch/~ihnatova/index.html) dataset
 - Adapt the paths in `paths.yml` if necessary

### Add Artificial Corruptions
Before you run the code, make sure that you added/edited the respective entries in the 
`paths.yml` file. The corrupted datasets can then be generated with:
```
python add_corruptions.py --dataset dataset 
                          --artifacts artifacts
                          --noise_std_dev None
                          --jpg_quality None
                          --resolution hr
```
To add corruptions, change the parameter for `noise_std_dev` or `jpg_quality`. 
We use a standard deviation of 8 for our experiments with sensor noise and 
a quality of 30 for our experiments with JPEG artifacts.
Use `python add_corruptions.py -h` to view optional arguments.
