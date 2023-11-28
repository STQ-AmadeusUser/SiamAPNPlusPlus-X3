import os
import argparse
import numpy as np
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
from torch.utils.data import DataLoader
from pysot.datasets.dataset_adapn_deploy import TrkDataset
from pysot.core.config_adapn import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SiamAPN++ Calibration Data')
    parser.add_argument('--cfg', type=str, default='../experiments/config.yaml',
                        help='configuration of tracking')
    args = parser.parse_args()
    args.calib_z = "./calibration/template/"
    args.calib_x = "./calibration/search/"
    args.calib_path = [args.calib_z, args.calib_x]
    return args


def calibration():
    # preprocess and configure
    args = parse_args()
    for calib_path in args.calib_path:
        if not os.path.exists(calib_path): os.makedirs(calib_path)
    cfg.merge_from_file(args.cfg)

    # build dataset
    train_dataset = TrkDataset()
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8,
                              pin_memory=True, sampler=None, drop_last=True)

    for iter_id, batch_data in enumerate(train_loader):
        template = batch_data['template']  # bx3x127x127
        search = batch_data['search']  # bx3x287x287

        print('template shape: ', template.shape)
        print('search shape: ', search.shape)

        if iter_id < 128:
            # z = np.transpose(template.squeeze(0).numpy().astype(np.int8), (1, 2, 0))
            z = template.numpy().astype(np.uint8)
            z.tofile(args.calib_path[0] + "z" + "_" + str(iter_id) + ".bin")
            # x = np.transpose(search.squeeze(0).numpy().astype(np.int8), (1, 2, 0))
            x = search.numpy().astype(np.uint8)
            x.tofile(args.calib_path[1] + "x" + "_" + str(iter_id) + ".bin")

        else:
            break


if __name__ == '__main__':
    calibration()
