import argparse
import os
import torch
import shutil
import yaml


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument(
            '--model_name',
            type=str,
            default='',
        )
        self.parser.add_argument(
            '--dataset_root_folder',
            type=str,
            default=
            '/labdata/jyuan/grasp_pcd_dataset',
            help='path to root directory of the pcd dataset.')
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=64,
                                 help='data batch size.')
        self.parser.add_argument('--num_grasps_per_object',
                                 type=int,
                                 default=64)
        self.parser.add_argument('--npoints',
                                 type=int,
                                 default=1000,
                                 help='number of points in point cloud')
        self.parser.add_argument(
            '--occlusion_angle_threshold',
            type=int,
            default=0,
            help=
            'clusters the points to nclusters to be selected for simulating the dropout'
        )
        self.parser.add_argument(
            '--occlusion_drop_rate',
            type=float,
            default=0.7,
            help=
            'number of points to occlude per cluster')
        self.parser.add_argument('--xy_depth_noise', type=float,
                                 default=0.0015)  # to be used in the data reader.
        
        self.parser.add_argument('--z_depth_noise', type=float,
                                 default=0.003)  # to be used in the data reader.
        
        self.parser.add_argument(
            '--gpu_ids',
            type=str,
            default='1,2,3',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir',
                                 type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument(
            '--serial_batches',
            action='store_true',
            help='if true, takes meshes in order, otherwise takes them randomly'
        )
        self.parser.add_argument('--seed',
                                 type=int,
                                 help='if specified, uses seed')
        self.parser.add_argument(
            '--gripper_pc_npoints',
            type=int,
            default=250,
            help=
            'number of points representing the gripper. -1 just uses the points on the finger and also the base. other values use subsampling of the gripper mesh'
        )
        self.parser.add_argument(
            '--model_scale',
            type=int,
            default=3,
            help=
            'Size of network, increments of 1 represent powers of 2. Size=3 is default, 2 halfs the size of all layers, 4 doubles it'
        )
        self.parser.add_argument(
            '--pointnet_radius',
            help='Radius for ball query for PointNet++, just the first layer',
            type=float,
            default=0.02)
        self.parser.add_argument(
            '--pointnet_nclusters',
            help=
            'Number of cluster centroids for PointNet++, just the first layer',
            type=int,
            default=128)
        self.parser.add_argument(
            '--init_type',
            type=str,
            default='normal',
            help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument(
            '--init_gain',
            type=float,
            default=0.02,
            help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--num_threads',
                                 default=4,
                                 type=int,
                                 help='# threads for loading data')



    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            name = 'evaluator'
            name += "_lr_" + str(self.opt.lr).split(".")[-1] + "_bs_" + str(
                self.opt.batch_size)
            name += "_scale_" + str(self.opt.model_scale) + "_npoints_" + str(
                self.opt.pointnet_nclusters) + "_radius_" + str(
                    self.opt.pointnet_radius).split(".")[-1]
            self.opt.name = name
            if self.opt.model_name == '':
                self.opt.name = name
            else:
                self.opt.name = self.opt.model_name
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if os.path.isdir(expr_dir) and not self.opt.continue_train:
                option = "Directory " + expr_dir + \
                    " already exists and you have not chosen to continue to train.\nDo you want to override that training instance with a new one the press (Y/N)."
                print(option)
                while True:
                    choice = input()
                    if choice.upper() == "Y":
                        print("Overriding directory " + expr_dir)
                        shutil.rmtree(expr_dir)
                        mkdir(expr_dir)
                        break
                    elif choice.upper() == "N":
                        print(
                            "Terminating. Remember, if you want to continue to train from a saved instance then run the script with the flag --continue_train"
                        )
                        return None
            else:
                mkdir(expr_dir)

            yaml_path = os.path.join(expr_dir, 'opt.yaml')
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(args, yaml_file)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
