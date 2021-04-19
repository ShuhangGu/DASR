import os
import argparse
import json
import yaml
from SRN.options.options import parse


def create_auto_json(dataset, artifact):
    options_json_path = './SRN/options/train/train_DASR_auto_reproduce_aim2019.json'

    with open('./paths.yml', 'r') as stream:
        PATHS = yaml.load(stream, Loader=yaml.FullLoader)


    with open(options_json_path, 'r+') as json_file:
        config = json.load(json_file)

        config['name'] = '0603_DASR_SRN_auto_reproduce_{}'.format(dataset)
        config['datasets']['train']['dataroot_HR'] = PATHS[dataset][artifact]['target']
        config['datasets']['train']['dataroot_fake_LR'] = '../../DSN_results/0603_DSN_LRs_{}/imgs_from_target'.format(dataset)
        config['datasets']['train']['dataroot_real_LR'] = PATHS[dataset][artifact]['source']
        config['datasets']['train']['dataroot_fake_weights'] = '../../DSN_results/0603_DSN_LRs_{}/ddm_target'.format(dataset)
        config['datasets']['val']['dataroot_HR'] = PATHS[dataset][artifact]['valid_hr']
        config['datasets']['val']['dataroot_LR'] = PATHS[dataset][artifact]['valid_lr']
        json_file.seek(0)  # rewind
        json.dump(config, json_file)
        json_file.truncate()  # if the new data is smaller than the previous


def main():
    parser = argparse.ArgumentParser(description='Auto Reproduce Script')
    parser.add_argument('--dataset', default=None, type=str,
                        help='set datasets in  (aim2019, realsr)')
    parser.add_argument('--artifact', default=None, type=str,
                        help='set artifact in  (tdsr, tdrealsr, tddiv2k)')
    opt = parser.parse_args()

    os.system('cd ./DSN; sh auto_reproduce_launcher_{}.sh'.format(opt.dataset))
    create_auto_json(opt.dataset, opt.artifact)
    os.system('cd ./SRN; python train.py -opt options/train/train_DASR_auto_reproduce_{}.json'.format(opt.dataset))


if __name__ == '__main__':
    main()