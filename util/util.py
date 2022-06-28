from __future__ import print_function
"""
@FileName: util.py
@Time    : 3/10/2021
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

"""This module contains simple helper functions """

import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F


def get_bb_from_mask(mask):
    coords = np.argwhere(mask)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    # print('xmin', x_min)

    return [x_min, y_min+1, x_max, y_max+1]


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  # post-processing: tranpose and scaling
        image_numpy = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path,quality=100) #added by Mia (quality)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def batch_local(output, bbs_local, pad):
    # out = torch.empty((1, output.size(1), output.size(2), output.size(3))).cuda()
    for i in range(output.size(0)):
        output_local = output[i, :, bbs_local[0][i]:bbs_local[1][i],
              bbs_local[2][i]:bbs_local[3][i]]

        output_pad = F.pad(output_local, pad=(pad[0][i], pad[1][i], pad[2][i], pad[3][i]))
        # print(out.size(), output_pad.size())
        # input()
        if i == 0:
            out = output_pad.unsqueeze(0)
        else:
            out = torch.cat((out, output_pad.unsqueeze(0)), 0)

    return out


##### RGB - YCbCr

# Helper for the creation of module-global constant tensors
def _t(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO inherit this
    # device = torch.device("cpu") # TODO inherit this
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)

# Helper for color matrix multiplication
def _mul(coeffs, image):
    # This is implementation is clearly suboptimal.  The function will
    # be implemented with 'einsum' when a bug in pytorch 0.4.0 will be
    # fixed (Einsum modifies variables in-place #7763).
    coeffs = coeffs.to(image.device)
    r0 = image[:, 0:1, :, :].repeat(1, 3, 1, 1) * coeffs[:, 0].view(1, 3, 1, 1)
    r1 = image[:, 1:2, :, :].repeat(1, 3, 1, 1) * coeffs[:, 1].view(1, 3, 1, 1)
    r2 = image[:, 2:3, :, :].repeat(1, 3, 1, 1) * coeffs[:, 2].view(1, 3, 1, 1)
    return r0 + r1 + r2
    # return torch.einsum("dc,bcij->bdij", (coeffs.to(image.device), image))

_RGB_TO_YCBCR = _t([[0.257, 0.504, 0.098], [-0.148, -0.291, 0.439], [0.439 , -0.368, -0.071]])
_YCBCR_OFF = _t([0.063, 0.502, 0.502]).view(1, 3, 1, 1)


def rgb2ycbcr(rgb):
    """sRGB to YCbCr conversion."""
    clip_rgb=False
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return _mul(_RGB_TO_YCBCR, rgb) + _YCBCR_OFF.to(rgb.device)


def ycbcr2rgb(rgb):
    """YCbCr to sRGB conversion."""
    clip_rgb=False
    rgb = _mul(torch.inverse(_RGB_TO_YCBCR), rgb - _YCBCR_OFF.to(rgb.device))
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return rgb


def compute_final(composite, output, mask):
    ycbcr_composite = rgb2ycbcr(composite)
    ycbcr_output = rgb2ycbcr(output)
    cbcr_composite = ycbcr_composite[:, 1:, :, :]
    y_output = ycbcr_output[:, 0, :, :]

    final = ycbcr2rgb(torch.cat([y_output.unsqueeze(1), cbcr_composite], dim=1))
    final = final * mask + (1-mask) * output

    return final


def get_l(rgb):
    ycbcr = rgb2ycbcr(rgb)
    y = ycbcr[:, 0, :, :]

    return y.unsqueeze(1)
