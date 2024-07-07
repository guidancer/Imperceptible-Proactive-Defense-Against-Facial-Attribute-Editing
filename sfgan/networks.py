import torch
import torch.nn as nn
from torch.nn import init
import functools
from sfgan.dwt_blocks import SFDWTBlock,SFIDWTBlock
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SFGeneratorNeo(nn.Module):
    def __init__(self,filter = 64):
        super(SFGeneratorNeo, self).__init__()
        self.sfencoder = SFEncoder(filter=filter)
        self.sfdecoder = SFDecoder(filter=filter)

    def forward(self,x):
        feature_encoded,mc_pool = self.sfencoder(x)
        pert_decoded = self.sfdecoder(feature_encoded,mc_pool)
        return pert_decoded


"""
Define the Encoder
"""
class SFEncoder(nn.Module):
    def __init__(self, filter = 64):
        super(SFEncoder, self).__init__()
        self.conv_2d_1 = SFConv2DBlock(input_channels=3, output_channels=int(filter/2), kernel_size=3, stride=1, padding=1)

        self.dwt_block_1 = SFDWTBlock(int(filter/2)).to(device)
        self.conv_2d_1_2 = SFConv2DBlock(input_channels=int(filter/2), output_channels=int(filter/2), kernel_size=4, stride=2, padding=1)
        self.conv_2d_2 = SFConv2DBlock(input_channels=filter, output_channels=filter, kernel_size=3, stride=1, padding=1)

        self.dwt_block_2 = SFDWTBlock(filter).to(device)
        self.conv_2d_1_3 = SFConv2DBlock(input_channels=int(filter/2), output_channels=filter, kernel_size=4, stride=4, padding=1)
        self.conv_2d_2_3 = SFConv2DBlock(input_channels=filter, output_channels=filter, kernel_size=4, stride=2, padding=1)
        self.conv_2d_3 = SFConv2DBlock(input_channels=filter*3, output_channels=filter*2, kernel_size=3, stride=1, padding=1)

        self.dwt_block_3 = SFDWTBlock(filter*2).to(device)
        self.conv_2d_1_4 = SFConv2DBlock(input_channels=int(filter/2), output_channels=filter*2, kernel_size=4, stride=8, padding=1)
        self.conv_2d_2_4 = SFConv2DBlock(input_channels=filter, output_channels=filter*2, kernel_size=4, stride=4, padding=1)
        self.conv_2d_3_4 = SFConv2DBlock(input_channels=filter*2, output_channels=filter*2, kernel_size=4, stride=2, padding=1)
        self.conv_2d_4 = SFConv2DBlock(input_channels=filter*2*4, output_channels=filter*4, kernel_size=3, stride=1, padding=1)

        self.dwt_block_4 = SFDWTBlock(filter*4).to(device)
        self.conv_2d_1_5 = SFConv2DBlock(input_channels=int(filter/2), output_channels=filter*4, kernel_size=4, stride=16, padding=1)
        self.conv_2d_2_5 = SFConv2DBlock(input_channels=filter, output_channels=filter*4, kernel_size=4, stride=8, padding=1)
        self.conv_2d_3_5 = SFConv2DBlock(input_channels=filter*2, output_channels=filter*4, kernel_size=4, stride=4, padding=1)
        self.conv_2d_4_5 = SFConv2DBlock(input_channels=filter*4, output_channels=filter*4, kernel_size=4, stride=2, padding=1)
        self.conv_2d_5 = SFConv2DBlock(input_channels=filter*4*5, output_channels=filter*8, kernel_size=3, stride=1, padding=1)

        self.dwt_block_5 = SFDWTBlock(filter*8).to(device)
        self.conv_2d_6 = SFConv2DBlock(input_channels=filter*8, output_channels=filter*8, kernel_size=4, stride=2, padding=1)
    

    def forward(self,x):
        """Multi Connections Pool"""
        mc_pool={}
        

        """Get the x from B*3*256*256 to B*32*256*256 to get x_1"""
        x_1 = self.conv_2d_1(x)
        """Get the x_1 from B*32*256*256 to 4*B*32*128*128 to get LL1,LH1,HL1,HH1"""
        LL1,LH1,HL1,HH1 = self.dwt_block_1(x_1)
        """Use the mc_pool to record high fres for skip connection"""
        mc_pool['high_fre_1'] = LH1,HL1,HH1
        

        """Get x_1 from B*32*256*256 to B*32*128*128 to get x_1_2"""
        x_1_2 = self.conv_2d_1_2(x_1)
        """Cat x_1_2 and LL1 to B*(32*2)*128*128 to get x_1_concat"""
        x_1_concat = torch.cat([x_1_2,LL1],dim=1)
        

        """Get the x_1_concat from B*(32*2)*128*128 to B*64*128*128 to get x_2 """
        x_2 = self.conv_2d_2(x_1_concat)
        """Get the x_2 from B*64*128*128 to 4*B*64*64*64 to get LL2,LH2,HL2,HH2"""
        LL2,LH2,HL2,HH2 = self.dwt_block_2(x_2)
        """Use the mc_pool to record high fres for skip connection"""
        mc_pool['high_fre_2'] = LH2,HL2,HH2


        """Get the x_2 from B*64*128*128 to B*64*64*64 to get x_2_3"""
        x_2_3 = self.conv_2d_2_3(x_2)
        """Get the x_1 from B*32*256*256 to B*64*64*64 to get x_1_3"""
        x_1_3 = self.conv_2d_1_3(x_1)
        """Cat x_2_3 x_1_3 and LL2 to B*(64*3)*64*64 to get x_2_concat"""
        x_2_concat = torch.cat([x_2_3,x_1_3,LL2],dim=1)


        """Get the x_2_concat from B*(64*3)*64*64 to B*128*64*64 to get x_3 """
        x_3 = self.conv_2d_3(x_2_concat)
        """Get the x_3 from B*128*64*64 to 4*B*128*32*32 to get LL3,LH3,HL3,HH3"""
        LL3,LH3,HL3,HH3 = self.dwt_block_3(x_3)
        """Use the mc_pool to record high fres for skip connection"""
        mc_pool['high_fre_3'] = LH3,HL3,HH3
    
        """Get the x_3 from B*128*64*64 to B*128*32*32 to get x_3_4"""
        x_3_4 = self.conv_2d_3_4(x_3) 
        """Get the x_2 from B*64*128*128 to B*128*32*32 to get x_2_4"""
        x_2_4 = self.conv_2d_2_4(x_2)
        """Get the x_1 from B*32*256*256 to B*128*32*32 to get x_1_4"""
        x_1_4 = self.conv_2d_1_4(x_1)
        """Cat x_3_4 x_2_4 x_1_4 and LL3 to B*(128*4)*32*32 to get x_3_concat"""
        x_3_concat = torch.cat([x_3_4,x_2_4,x_1_4,LL3],dim=1)

        """Get the x_3_concat from B*(128*4)*32*32 to B*256*32*32"""
        x_4 = self.conv_2d_4(x_3_concat)
        """Get the x_4 from B*256*32*32 to 4*B*256*16*16 to get LL4,LH4,HL4,HH4"""
        LL4,LH4,HL4,HH4 = self.dwt_block_4(x_4)
        """Use the mc_pool to record high fres for skip connection"""
        mc_pool['high_fre_4'] = LH4,HL4,HH4

        """Get the x_4 from B*256*32*32 to B*256*16*16 to get x_4_5"""
        x_4_5 = self.conv_2d_4_5(x_4)
        """Get the x_3 from B*128*64*64 to B*256*16*16 to get x_3_5"""
        x_3_5 = self.conv_2d_3_5(x_3)
        """Get the x_2 from B*64*128*128 to B*256*16*16 to get x_2_5"""
        x_2_5 = self.conv_2d_2_5(x_2)
        """Get the x_1 from B*32*256*256 to B*256*16*16 to get x_1_5"""
        x_1_5 = self.conv_2d_1_5(x_1)
        """Cat x_4_5 x_3_5 x_2_5 x_1_5 and LL4 to B*(256*5)*16*16 to get x_4_concat"""
        x_4_concat = torch.cat([x_4_5,x_3_5,x_2_5,x_1_5,LL4],dim=1)
        
        """Get x_4_concat from B*(256*5)*16*16 to B*512*16*16 to get x_5"""
        x_5 = self.conv_2d_5(x_4_concat)
        
        LL5,LH5,HL5,HH5 = self.dwt_block_5(x_5)
        mc_pool['high_fre_5'] = LH5,HL5,HH5
        x_6 = self.conv_2d_6(x_5)
        x_encoded = x_6+LL5
        return x_encoded,mc_pool


"""
Define the Decoder Used in SFGAN
"""
class SFDecoder(nn.Module):
    def __init__(self, filter = 64):
        super(SFDecoder,self).__init__()
        self.weight_fusion=0.5
        self.idwt_block_5 = SFIDWTBlock(filter*8).to(device)
        self.up_path_block5 = SFConvTranspose2DBlock(input_channels=filter*8, output_channels=filter*4)
        self.high_path_block5 = SFConv2DBlock(input_channels=filter*8, output_channels=filter*4, kernel_size=3, stride=1, padding=1, activation='ReLU')

        self.idwt_block_4 = SFIDWTBlock(filter*4).to(device)
        self.up_path_block4 = SFConvTranspose2DBlock(input_channels=filter*4, output_channels=filter*2)
        self.high_path_block4 = SFConv2DBlock(input_channels=filter*4, output_channels=filter*2, kernel_size=3, stride=1, padding=1, activation='ReLU')
        
        self.idwt_block_3 = SFIDWTBlock(filter*2).to(device)
        self.up_path_block3 = SFConvTranspose2DBlock(input_channels=filter*2, output_channels=filter*1)
        self.high_path_block3 = SFConv2DBlock(input_channels=filter*2, output_channels=filter, kernel_size=3, stride=1, padding=1, activation='ReLU')

        self.idwt_block_2 = SFIDWTBlock(filter).to(device)
        self.up_path_block2 = SFConvTranspose2DBlock(input_channels=filter*1, output_channels=int(filter/2))
        self.high_path_block2 = SFConv2DBlock(input_channels=filter, output_channels=int(filter/2), kernel_size=3, stride=1, padding=1, activation='ReLU')

        self.idwt_block_1 = SFIDWTBlock(int(filter/2)).to(device)
        self.up_path_block1 = SFConvTranspose2DBlock(input_channels=int(filter/2), output_channels=3, norm='none', activation='Tanh')
        self.high_path_block1 = SFConv2DBlock(input_channels=int(filter/2), output_channels=3, kernel_size=3, stride=1, padding=1, norm='none', activation='Tanh')
        

    def forward(self,x6,mc_pool):
        LH5,HL5,HH5 = mc_pool['high_fre_5']
        LH4,HL4,HH4 = mc_pool['high_fre_4']
        LH3,HL3,HH3 = mc_pool['high_fre_3']
        LH2,HL2,HH2 = mc_pool['high_fre_2']
        LH1,HL1,HH1 = mc_pool['high_fre_1']

        """Construct HF From (512,8,8)*3 to (512,16,16)"""
        x5_decoded = self.idwt_block_5(LL = x6, LH = LH5, HL = HL5, HH = HH5)
        """(512,16,16) to (256,16,16)"""
        x5 = self.up_path_block5(x6)
        x5_decoded_rf = self.high_path_block5(x5_decoded)

        """Construct HF From (256,16,16)*3 to (256,32,32)"""
        x4_decoded = self.idwt_block_4(LL = x5_decoded_rf, LH = LH4, HL = HL4, HH = HH4)
        """X From (256,32,32) to (128,32,32)"""
        x4 = self.up_path_block4(x5)
        x4_decoded_rf = self.high_path_block4(x4_decoded)

        """Construct HF From (128,32,32)*3 to (128,64,64)"""
        x3_decoded = self.idwt_block_3(LL = x4_decoded_rf, LH = LH3, HL = HL3, HH = HH3)
        """X From (128,64,64) to (64,64,64)"""
        x3 = self.up_path_block3(x4)
        x3_decoded_rf = self.high_path_block3(x3_decoded)

        """Construct HF From (64,64,64)*3 to (64,128,128)"""
        x2_decoded = self.idwt_block_2(LL = x3_decoded_rf, LH = LH2, HL = HL2, HH = HH2)
        """X From (64,128,128) to (32,128,128)"""
        x2 = self.up_path_block2(x3)
        x2_decoded_rf = self.high_path_block2(x2_decoded)

        """Construct HF From (32,128,128)*3 to (32,256,256)"""
        x1_decoded = self.idwt_block_1(LL = x2_decoded_rf, LH = LH1, HL = HL1, HH = HH1)
        """X From (32,256,256) to (3,256,256)"""
        x1 = self.up_path_block1(x2)
        x1_decoded_rf = self.high_path_block1(x1_decoded)
        final_out = self.weight_fusion*x1+(1-self.weight_fusion)*x1_decoded_rf
        return final_out


"""
Define The Conv2D Block 
"""
class SFConv2DBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, activation = 'LeakyReLU', norm = 'instance', bias = True):
        super(SFConv2DBlock,self).__init__()
        """Define Conv 2D Blocks for SFGAN"""
        self.conv_2d = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.norm = norm 

        if norm == 'instance':
            """affine=False The Parameters of the Norm Layer do not update"""
            self.norm_layer = nn.InstanceNorm2d(output_channels)    
        elif norm == 'batch':
            self.norm_layer = nn.BatchNorm2d(output_channels)
        
        if activation == 'LeakyReLU':
            self.activate_func = nn.LeakyReLU(0.2, False)   
        elif activation == 'ReLU':
            self.activate_func = nn.ReLU()
        elif activation == 'GeLU':
            self.activate_func = nn.GELU()
        elif activation == 'Tanh':
            self.activate_func = nn.Tanh()

    def forward(self, x):
        x = self.conv_2d(x)
        if self.norm != 'none':
            x = self.norm_layer(x)
        x = self.activate_func(x)
        return x
    
"""
Define The ConvTranspose2DBlock
"""
class SFConvTranspose2DBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, activation = 'ReLU', norm = 'instance', bias = True):
        super(SFConvTranspose2DBlock,self).__init__()
        """Define Conv 2D Blocks for SFGAN"""
        self.trans_2d = nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.norm = norm

        if norm == 'instance':
            """affine=False The Parameters of the Norm Layer do not update"""
            self.norm_layer = nn.InstanceNorm2d(output_channels)    
        elif norm == 'batch':
            self.norm_layer = nn.BatchNorm2d(output_channels)
        
        if activation == 'LeakyReLU':
            self.activate_func = nn.LeakyReLU(0.2, False)   
        elif activation == 'ReLU':
            self.activate_func = nn.ReLU()
        elif activation == 'GeLU':
            self.activate_func = nn.GELU()
        elif activation == 'Tanh':
            self.activate_func = nn.Tanh()

    def forward(self,x):
        x = self.trans_2d(x)
        if self.norm != 'none':
            x = self.norm_layer(x)
        x = self.activate_func(x)
        return x


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

def GetSFG(init_type,init_gain,gpu_ids=[], filter = 64):
    net = SFGeneratorNeo(filter = filter)
    return init_net(net, init_type, init_gain, gpu_ids)


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
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def GetSFD(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
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

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


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
    
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    # 设定了不同的GAN目标
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        # 设置了一个缓存区？存放真实标签和虚假标签
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "fsgan":
            self.loss = nn.BCELoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        #创建和输入一样大小的标签张量
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
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
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
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None