import os
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from sfgan.networks import GetSFG,GetSFD,GANLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SFGANNet(nn.Module):
    def __init__(self,options, filter=64):
        super(SFGANNet, self).__init__() 
        """Set Generator and Discriminator"""
        self.netG = GetSFG('normal', 0.02, filter = filter).to(device)
        self.netD = GetSFD(3, 64, 'basic',3, 'instance', 'normal', 0.02).to(device)

        """
        Set Criterion For GAN loss
        """
        if options.gan_loss_type=='lsgan':
            self.gan_loss=GANLoss("lsgan").to(device) #use lsgan for stable training if this and vgg+L1 then the lambda_G_GAN should be 0.001
        elif options.gan_loss_type=='vanilla':
            self.gan_loss=GANLoss("vanilla").to(device) #use vanilla
        elif options.gan_loss_type=='fsgan':
            self.gan_loss=GANLoss("fsgan").to(device) #use when sigmoid as output

        """
        Set Optimizers
        """
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=options.lr, betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=options.lr, betas=(0.5, 0.999))
        """
        Set Schedulers if needed
        """
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
