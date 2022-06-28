"""
@FileName: cvictim_dataset.py
@Time    : 3/10/2021
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import os.path
import torch

from data.base_dataset import BaseDataset, data_augmentation, get_transform
from PIL import Image
import glob
import imgaug as ia

import numpy as np
import torchvision.transforms as transforms
import platform
from util import util
import torch.nn.functional as F


class CvictimDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values

        if platform.system().lower() == 'windows':
            parser.set_defaults(dataset_root='path\\to\\Victimdata')
        else:
            pass
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root

        BaseDataset.__init__(self, opt)
        self.totensor = transforms.ToTensor()
        self.isTrain = opt.isTrain

        if opt.isTrain:
            real_folder = os.path.join(opt.dataset_root, 'real/*.jpg')
            composite_folder = os.path.join(opt.dataset_root, 'composite/*.jpg')
            mask_folder = os.path.join(opt.dataset_root, 'mask/*.png')
            # reald_folder = os.path.join(opt.dataset_root, 'realc/*.jpg')

        elif not opt.isTrain:
            real_folder = os.path.join(opt.dataset_root, 'real/*.jpg')
            composite_folder = os.path.join(opt.dataset_root, 'composite/*.jpg')
            mask_folder = os.path.join(opt.dataset_root, 'mask/*.png')
        self.real_list = sorted(glob.glob(real_folder))
        self.composite_list = sorted(glob.glob(composite_folder))
        self.mask_list = sorted(glob.glob(mask_folder))
        # self.reald_list = sorted(glob.glob(reald_folder))
        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.real_list]

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        try:
            name = self.image_name[index]
            path = self.real_list[index]

            # load image
            real = np.array(Image.open(self.real_list[index]))
            composite = np.array(Image.open(self.composite_list[index]))
            mask = np.array(Image.open(self.mask_list[index]))

            mask[mask != 0] = 255
            mask = mask.astype(np.uint8)

            assert mask.ndim == 2
            [x_min, y_min, x_max, y_max] = util.get_bb_from_mask(mask)

            bbs = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=y_min, y1=x_min, x2=y_max, y2=x_max)
            ], shape=composite.shape)

            composite, real, mask, bbs = data_augmentation(self.opt, composite, mask, bbs, real=real)
            real = self.totensor(real.copy())
            composite = self.totensor(composite.copy())
            mask = self.totensor(mask.copy()).float()
            bbs2 = bbs.deepcopy().bounding_boxes[0]


            local_x1 = max(0, bbs2.x1_int - 60)
            local_x2 = min(self.opt.crop_size, bbs2.x2_int + 60)
            local_y1 = max(0, bbs2.y1_int - 60)
            local_y2 = min(self.opt.crop_size, bbs2.y2_int + 60)

            pad_left = (self.opt.crop_size - (local_x2 - local_x1)) // 2
            pad_right = self.opt.crop_size - (local_x2 - local_x1 + pad_left)
            pad_top = (self.opt.crop_size - (local_y2 - local_y1)) // 2
            pad_bottom = self.opt.crop_size - (local_y2 - local_y1 + pad_top)

            real_local = real.clone()[:, local_y1:local_y2, local_x1:local_x2]
            composite_local = composite.clone()[:, local_y1:local_y2, local_x1:local_x2]
            mask_local = mask.clone()[:, local_y1:local_y2, local_x1:local_x2]

            real_local = F.pad(real_local, pad=(pad_left, pad_right, pad_top, pad_bottom))
            composite_local = F.pad(composite_local, pad=(pad_left, pad_right, pad_top, pad_bottom))
            mask_local = F.pad(mask_local, pad=(pad_left, pad_right, pad_top, pad_bottom))

            inputs = torch.cat([composite, mask], dim=0)

            return {'inputs': inputs,
                    'composite': composite, 'composite_local': composite_local,
                    'real': real, 'real_local': real_local,
                    'mask': mask, 'mask_local': mask_local,
                    'bbs': [bbs2.y1_int, bbs2.y2_int, bbs2.x1_int, bbs2.x2_int],
                    'bbs_local': [local_y1, local_y2, local_x1, local_x2],
                    'pad': [pad_left, pad_right, pad_top, pad_bottom],
                    'name': name, 'img_path': path}

        except Exception as e:
            print(inputs.size(), composite.size(), real.size(), mask.size(), name, path)

    def __len__(self):
        """Return the total number of images."""
        return len(self.real_list)
