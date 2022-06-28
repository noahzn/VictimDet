"""
@FileName: networks.py
@Time    : 3/8/2021
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from torchvision import transforms
import os


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unetatt':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=True)

    elif netG == 'unettwo':
        net = TwoBranchesUnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'decomp':
        net = DecompositionNet(input_nc=3, output_nc=4)

        # input(net)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], use_attention=False):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_attention=use_attention)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_attention=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=False)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=False)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=True)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        # input(self.model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input)


def unet_downconv(in_c, out_c, norm_layer):

    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(4, 4), stride=(2, 2), padding=1),
        norm_layer(in_c),
        nn.LeakyReLU(0.2, True),
    )

def unet_upconv(in_c, out_c, norm_layer):

    # return nn.Sequential(
    #     nn.ConvTranspose2d(in_c, out_c, kernel_size=(4, 4), stride=(2, 2), padding=1),
    #     norm_layer(in_c),
    #     nn.ReLU(True),
    # )

    return nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=0),
        # nn.Conv2d(in_c, out_c, kernel_size=1),
        norm_layer(in_c),
        nn.ReLU(True),
    )




class TwoBranchesUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(TwoBranchesUnetGenerator, self).__init__()
        # construct unet structure
        self.down1 = nn.Conv2d(input_nc, ngf, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.down2 = unet_downconv(ngf, ngf*2, norm_layer=norm_layer)
        self.down3 = unet_downconv(ngf*2, ngf*4, norm_layer=norm_layer)
        self.down4 = unet_downconv(ngf*4, ngf*8, norm_layer=norm_layer)
        self.down5 = unet_downconv(ngf*8, ngf*8, norm_layer=norm_layer)
        self.down6 = unet_downconv(ngf*8, ngf*8, norm_layer=norm_layer)
        self.down7 = unet_downconv(ngf*8, ngf*8, norm_layer=norm_layer)
        self.down8 = nn.Conv2d(ngf*8, ngf*8, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.relu = nn.ReLU(True)

        self.up8_2 = unet_upconv(ngf*8, ngf*8, norm_layer=norm_layer)
        self.up7_2 = unet_upconv(ngf*16, ngf*8, norm_layer=norm_layer)
        self.up6_2 = unet_upconv(ngf*16, ngf*8, norm_layer=norm_layer)
        self.up5_2 = unet_upconv(ngf*16, ngf*8, norm_layer=norm_layer)
        self.up4_2 = unet_upconv(ngf*16, ngf*4, norm_layer=norm_layer)
        self.up3_2 = unet_upconv(ngf*8, ngf*2, norm_layer=norm_layer)
        self.up2_2 = unet_upconv(ngf*4, ngf, norm_layer=norm_layer)
        self.up1_2 = nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.up8 = unet_upconv(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.up7 = unet_upconv(ngf * 16, ngf * 8, norm_layer=norm_layer)
        self.up6 = unet_upconv(ngf * 16, ngf * 8, norm_layer=norm_layer)
        self.up5 = unet_upconv(ngf * 16, ngf * 8, norm_layer=norm_layer)
        self.up4 = unet_upconv(ngf * 16, ngf * 4, norm_layer=norm_layer)
        self.up3 = unet_upconv(ngf * 8, ngf * 2, norm_layer=norm_layer)
        self.up2 = unet_upconv(ngf * 4, ngf, norm_layer=norm_layer)
        self.up1 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.tanh = nn.Tanh()


    def forward(self, x):
        """Standard forward"""
        down1 = self.leakyrelu(self.down1(x))
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        down8 = self.relu(self.down8(down7))

        up8 = self.up8(down8)
        up7 = self.up7(torch.cat([down7, up8], dim=1))
        up6 = self.up6(torch.cat([down6, up7], dim=1))
        up5 = self.up5(torch.cat([down5, up6], dim=1))
        up4 = self.up4(torch.cat([down4, up5], dim=1))
        up3 = self.up3(torch.cat([down3, up4], dim=1))
        up2 = self.up2(torch.cat([down2, up3], dim=1))
        up1 = self.tanh(self.up1(torch.cat([down1, up2], dim=1)))

        up8_ = self.up8_2(down8)
        up7_ = self.up7_2(torch.cat([down7, up8_], dim=1))
        up6_ = self.up6_2(torch.cat([down6, up7_], dim=1))
        up5_ = self.up5_2(torch.cat([down5, up6_], dim=1))
        up4_ = self.up4_2(torch.cat([down4, up5_], dim=1))
        up3_ = self.up3_2(torch.cat([down3, up4_], dim=1))
        up2_ = self.up2_2(torch.cat([down2, up3_], dim=1))
        up1_ = self.tanh(self.up1(torch.cat([down1, up2_], dim=1)))

        return up1, up1_

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up


        self.use_attention = use_attention

        if use_attention and not self.innermost:
            self.channel_attn = Channel_Attn(outer_nc+input_nc)

        elif use_attention and self.innermost:
            self.pixel_attn = Pixel_Attn(outer_nc + input_nc)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            ret = torch.cat([x, self.model(x)], 1)
            if self.use_attention and self.innermost:
                return self.pixel_attn(ret)[0]
            elif self.use_attention and not self.innermost:
                # return self.attention(ret) * ret
                return self.channel_attn(ret)[0]
                # return self.pixel_attn(ret)[0] + self.channel_attn(ret)[0]
            return ret


class Pixel_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Pixel_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # print(m_batchsize, C, width, height)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Channel_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Channel_Attn, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X C X C
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, C, -1)  # B X C X (W X H)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # B X (*W*H) x C
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)  # BX (N) X (N)
        proj_value = x.view(m_batchsize, C, -1)  # B X C X (W X H)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_attention=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        self.use_attention = use_attention

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if self.use_attention:
            self.attn1 = Self_Attn(512, 'relu')

        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        last = [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.last = nn.Sequential(*last)

    def forward(self, x):
        """Standard forward."""
        if self.use_attention:
            out = self.model(x)
            # input(out.size())
            out, p = self.attn1(out)
            return self.last(out)

        return self.last(self.model(x))


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class VGG16(nn.Module):
    """
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): 'conv2_1' Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): 'conv3_1' Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): 'conv4_1' Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): 'conv5_1'Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    """
    def __init__(self):
        super().__init__()
        features = list(models.vgg16(pretrained=True).features[:30])

        self.features = nn.ModuleList(features).eval().cuda()

    def forward(self, x, opt):
        content = []
        style = []
        with torch.no_grad():
            for ii, model in enumerate(self.features):
                x = model(x)
                if ii in opt.vgg_choose_content:
                    content.append(x)
                if ii in opt.vgg_choose_style:
                    style.append(x)

        return content, style


def vgg_preprocess(batch, opt):

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # batch = normalize(batch)
    y_batch = 0.299*batch[:, 0, :, :]+0.587*batch[:, 1, :, :]+0.114*batch[:, 2, :, :]

    tensortype = type(batch.data)
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std = tensortype(batch.data.size()).cuda()
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch = batch.sub(Variable(mean))
    batch = batch.div(Variable(std))

    return batch, torch.stack((y_batch, y_batch, y_batch), -3)


# Calculate Gram matrix (G = FF^T)
def gram_matrix(x):
    """
    Add gram matrix calculation form Perceptual Loss paper
    """
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        # self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, output_local, composite_local, target, mask=1, luminance=False):
        # print(img_local.size(), target_local.size(), target.size())
        img_local_vgg, img_local_vgg_l = vgg_preprocess(output_local, self.opt)
        composite_local_vgg, composite_local_vgg_l = vgg_preprocess(composite_local, self.opt)
        target_vgg, target_vgg_l = vgg_preprocess(target, self.opt)

        img_local_c, img_local_s = vgg(img_local_vgg, self.opt)
        img_local_l_c, img_local_l_s = vgg(img_local_vgg_l, self.opt)
        composite_local_c, composite_local_s = vgg(composite_local_vgg, self.opt)
        target_c, target_s = vgg(target_vgg, self.opt)
        target_l_c, target_l_s = vgg(target_vgg_l, self.opt)


        STYLE_WEIGHT = self.opt.vgg_style_weight
        CONTENT_WEIGHT = self.opt.vgg_content_weight
        TV_WEIGHT = self.opt.vgg_tv_weight

        # content loss
        content_loss = 0.0
        for i in range(len(img_local_c)):
            content_loss += CONTENT_WEIGHT * F.mse_loss(img_local_c[i], composite_local_c[i])

        content_loss /= len(img_local_c)
        # style loss

        for i in range(len(img_local_s)):
            img_local_s[i] *= F.interpolate(mask, [img_local_s[i].size(2), img_local_s[i].size(3)])
        img_local_gram = [gram_matrix(fmap) for fmap in img_local_s]
        target_gram = [gram_matrix(fmap) for fmap in target_s]

        style_loss = 0.0
        for feature_idx in range(len(img_local_gram)):
            # style_loss += torch.mean((img_local_gram[feature_idx] - target_gram[feature_idx]) ** 2)
            style_loss += F.mse_loss(img_local_gram[feature_idx], target_gram[feature_idx])

        style_loss /= len(img_local_gram)

        # TV loss
        diff_i = torch.sum(torch.abs(output_local[:, :, :, 1:] - output_local[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(output_local[:, :, 1:, :] - output_local[:, :, :-1, :]))
        tv_loss = TV_WEIGHT * (diff_i + diff_j)

        return CONTENT_WEIGHT * content_loss, STYLE_WEIGHT * style_loss, TV_WEIGHT * tv_loss

        # if self.opt.no_vgg_instance:
        #     return torch.mean((img_local_fea - target_fea) ** 2)
        # else:
        #     n, c, h, w = target_fea[0].size()
        #     img_local_fea_up = F.interpolate(img_local_fea[0], size=(h, w), mode='bilinear')
        #     return torch.mean((self.instancenorm(img_local_fea_up) - self.instancenorm(target_fea[0])) ** 2)


class DecompositionNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=4, pretrained=False):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv_2_ = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)

        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        c1 = F.relu(self.conv_1(img))
        c2 = F.relu(self.conv_2(c1))
        # c2_ = self.conv_2_(c1)
        c3 = F.relu(self.conv_3(c2))
        c4 = F.relu(self.conv_4(c3))

        c5 = F.relu(self.conv_5(c4))
        c6 = F.relu(self.conv_6(torch.cat((c2, c5), dim=1)))
        c7 = self.conv_7(torch.cat((c1, c6), dim=1))
        c8 = self.conv_8(c7)

        c9 = torch.sigmoid(c8)

        # print(c1.size(), c2.size(), c3.size(), c4.size(), c5.size(), c6.size(), c7.size(), c8.size())
        # input()
        reflectance, illumination = c9[:, :3, :, :], c9[:, 3:4, :, :]
        # reflectance = torch.sigmoid(c8[:, :3, :, :])
        # illumination = c8[:, 3:4, :, :]

        return illumination, reflectance


class LossRetinex(nn.Module):
    def __init__(self):
        super(LossRetinex, self).__init__()


    def gradient(self, input_tensor, direction):
        smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, r, i):
        r = 0.299 * r[:, 0, :, :] + 0.587 * r[:, 1, :, :] + 0.114 * r[:, 2, :, :]
        r = torch.unsqueeze(r, dim=1)
        return torch.mean(self.gradient(i, "x") * torch.exp(-10 * self.ave_gradient(r, "x")) +
                          self.gradient(i, "y") * torch.exp(-10 * self.ave_gradient(r, "y")))

    def smooth_r(self, r):
        r = 0.299 * r[:, 0, :, :] + 0.587 * r[:, 1, :, :] + 0.114 * r[:, 2, :, :]
        r = torch.unsqueeze(r, dim=1)
        return torch.mean(self.ave_gradient(r, "x") + self.ave_gradient(r, "y"))


    def smooth_i(self, d, i):
        # d = torch.unsqueeze(d, dim=1)
        return torch.mean(self.gradient(i, "x") * torch.exp(-10 * self.ave_gradient(d, "x")) +
                          self.gradient(i, "y") * torch.exp(-10 * self.ave_gradient(d, "y")))

    def recon(self, r, i, s):
        return F.l1_loss(r * i, s)

    def init_illumination_loss(self, R, I):
        km = torch.mean(R, dim=1)


        return F.l1_loss(km, I)

    def max_rgb_loss(self, image, illumination):
        n, c, h, w = image.size()
        max_rgb, _ = torch.max(image, 1)
        max_rgb = max_rgb.unsqueeze(1)

        # return F.l1_loss(illumination, max_rgb)
        return torch.norm(illumination-max_rgb, 1)/(n*c*h*w)


class SRMWrapper(torch.nn.Module):
    """Wraps a base model for Steganalysis Rich Model (SRM)-based noise analysis."""

    def __init__(self, input_channels: int = 3):
        """Creates a SRM analysis layer and prepares internal params."""
        # note: the base model should expect to process 3 extra channels in its inputs!
        super().__init__()
        self.input_channels = input_channels
        self.srm_conv = setup_srm_layer(input_channels)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Adds a stack of noise channels to the input tensor, and processes it using the base model."""
        # simply put, this is an early fusion of noise features...
        noise = self.srm_conv(img)
        return torch.cat([img, noise], dim=1)


def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0.,  1., -2.,  1.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0., -1.,  2., -1.,  0.],  # noqa: E241,E201
            [ 0.,  2., -4.,  2.,  0.],  # noqa: E241,E201
            [ 0., -1.,  2., -1.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1.,  2., -2.,  2., -1.],  # noqa: E241,E201
            [ 2., -6.,  8., -6.,  2.],  # noqa: E241,E201
            [-2.,  8., -12., 8., -2.],  # noqa: E241,E201
            [ 2., -6.,  8., -6.,  2.],  # noqa: E241,E201
            [-1.,  2., -2.,  2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights)
    return conv


def to_gray(batch):
    y_batch = 0.299 * batch[:, 0, :, :] + 0.587 * batch[:, 1, :, :] + 0.114 * batch[:, 2, :, :]

    return y_batch.unsqueeze(1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open('2.jpg')
    ts = transforms.ToTensor()
    img = ts(img)
    img = img.view(1, 3, img.size(1), img.size(2))
    # input(img.size())
    vgg = VGG16()
    feat = vgg(img, 'opt')

    # print(feat[0].size())

    feat = feat[0].view(-1, 256, 140, 210).cpu().detach().numpy()
    plt.figure()
    plt.imshow(feat[0][0, :, :])
    print(feat[0][0, :, :])
    for i in range(256):
        cv2.imwrite('{0}_0.jpg'.format(i), feat[0][i, :, :]*255)


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, output, target, mask, reduction='mean', beta=1.0):

        if beta < 1e-5:
            loss = torch.abs((output-target)*(1-mask))

        else:
            n = torch.abs(output-target)
            cond = n < beta
            loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
            loss = torch.abs(loss*(1-mask))

        if reduction == 'mean':
            return torch.sum(loss) / torch.sum(torch.abs(1-mask))
        else:
            return torch.sum(loss)


