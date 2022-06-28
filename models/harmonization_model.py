"""
@FileName: uns_model.py
@Time    : 6/3/2021
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import functools
from torch.nn.utils import spectral_norm
from util.util import batch_local, compute_final, get_l


class harmonizationmodel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='simbp')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--gp_ratio', type=float, default=0.1, help='weight for gradient_penalty')
            parser.add_argument('--lambda_d1', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--lambda_d2', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--lambda_v', type=float, default=1.0, help='weight for verification loss')
            parser.add_argument('--lambda_c', type=float, default=10.0, help='weight for color loss')
            parser.add_argument('--lambda_vgg', type=float, default=2.0, help='weight for vgg loss')
        # parser.set_defaults(no_vgg_instance=True)
        return parser

    def __init__(self, opt):
        """Initialize the DoveNet class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_SmoothL1', 'D1_real', 'D1_fake', 'D2_real', 'D2_fake','G_global', 'G_perceptional', 'content', 'style', 'tv']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D1', 'D2']
            self.visual_names = ['composite', 'real', 'output', 'final', 'real_local', 'output_local', 'mask',
                                 ]
        else:  # during test time, only load G
            self.model_names = ['G']
            self.visual_names = ['composite', 'real', 'output', 'real_local', 'output_local', 'mask']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()

        # define vgg network
        self.vgg = networks.VGG16()
        self.srm = networks.SRMWrapper().cuda()
        # input(self.vgg)

        if self.isTrain:
            self.gan_mode = opt.gan_mode

            netD1 = networks.define_D(3, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_attention=False)
            self.netD1 = networks.init_net(netD1, opt.init_type, opt.init_gain, self.gpu_ids)

            netD2 = networks.define_D(3, opt.ndf, opt.netD, opt.n_layers_D, opt.norm)
            self.netD2 = networks.init_net(netD2, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.perceptional_loss = networks.PerceptualLoss(opt)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSmoothL1 = networks.MaskedSmoothL1Loss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr * opt.g_lr_ratio,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr * opt.d_lr_ratio,
                                                 betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr * opt.d_lr_ratio,
                                                 betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)
            self.iter_cnt = 0

    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.inputs = inputs['inputs'].to(self.device)
        self.composite_local = inputs['composite_local'].to(self.device)
        self.composite = inputs['composite'].to(self.device)
        self.real = inputs['real'].to(self.device)
        self.real_local = inputs['real_local'].to(self.device)
        self.mask = inputs['mask'].to(self.device)
        self.mask_local = inputs['mask_local'].to(self.device)
        self.bbs = inputs['bbs']
        self.bbs_local = inputs['bbs_local']
        self.name = inputs['name']
        self.pad = inputs['pad']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.output = self.netG(self.inputs)

        output = self.netG(self.composite)
        self.output = output

        self.output_local = batch_local(self.output, self.bbs_local, self.pad)

    def backward_D1(self):
        """Calculate GAN loss for the discriminator"""
        # Fake;
        fake_AB = self.output
        pred_fake = self.netD1(fake_AB.detach())
        # input(pred_fake.size())

        if self.gan_mode == 'wgangp':
            global_fake = self.relu(1 + pred_fake).mean()
        else:
            global_fake = self.criterionGAN(pred_fake, False)

        self.loss_D1_fake = global_fake

        # Real
        real_AB = self.real
        pred_real = self.netD1(real_AB)
        if self.gan_mode == 'wgangp':
            global_real = self.relu(1 - pred_real).mean()
        else:
            global_real = self.criterionGAN(pred_real, True)

        self.loss_D1_real = global_real

        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real)# + self.opt.gp_ratio * self.loss_D1_gp)
        self.loss_D1.backward(retain_graph=True)

    def backward_D2(self):
        """Calculate GAN loss for the discriminator"""
        # Fake;
        fake_AB = self.output_local
        pred_fake = self.netD2(fake_AB.detach())
        if self.gan_mode == 'wgangp':
            global_fake = self.relu(1 + pred_fake).mean()
        else:
            global_fake = self.criterionGAN(pred_fake, False)

        self.loss_D2_fake = global_fake

        # Real
        real_AB = self.real_local
        pred_real = self.netD2(real_AB)
        if self.gan_mode == 'wgangp':
            global_real = self.relu(1 - pred_real).mean()
        else:
            global_real = self.criterionGAN(pred_real, True)

        self.loss_D2_real = global_real

        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) # + self.opt.gp_ratio * self.loss_D2_gp)
        self.loss_D2.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = self.output
        fake_AB_local = self.output_local

        # pred_fake = self.netD1(get_l(fake_AB))
        pred_fake = self.netD1(fake_AB)
        pred_fake_local = self.netD2(fake_AB_local)
        self.loss_G_global = self.criterionGAN(pred_fake, True)

        real_AB = self.real

        self.loss_G_GAN = self.opt.lambda_d1 * self.loss_G_global

        self.loss_G_SmoothL1 = self.criterionSmoothL1(self.output, self.composite, self.mask) * self.opt.lambda_L1

        self.loss_content, self.loss_style, self.loss_tv = self.perceptional_loss.compute_vgg_loss(self.vgg, self.output_local,
                                                                           self.composite_local,
                                                                           self.real_local, mask=self.mask_local,
                                                                           luminance=False)
        self.loss_G_perceptional = (self.loss_content + self.loss_style + self.loss_tv) * self.opt.lambda_vgg

        self.loss_G = self.loss_G_GAN + self.loss_G_perceptional + self.loss_G_SmoothL1

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()
        # update D1
        self.set_requires_grad(self.netD1, True)  # enable backprop for D
        self.optimizer_D1.zero_grad()  # set D's gradients to zero
        self.backward_D1()  # calculate gradients for D
        self.optimizer_D1.step()  # update D's weights
        # update D2
        self.set_requires_grad(self.netD2, True)  # enable backprop for D
        self.optimizer_D2.zero_grad()  # set D's gradients to zero
        self.backward_D2()  # calculate gradients for D
        self.optimizer_D2.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD1, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD2, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
