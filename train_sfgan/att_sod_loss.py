import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import numpy as np
import json
import kornia
import argparse
from argparse import ArgumentParser
from tqdm import tqdm
from utils.seed_utils import set_seed
from torch.utils.data import DataLoader
from sfgan.sfnet import SFGANNet
from utils.celeba_dataset import AttributeDataCelebA as CelebA
from models.attgan_model import AttModel
from skimage.metrics import structural_similarity
from face_sod_u2net.u2netsod import U2NetSODModel

"""Get AttGAN Args"""
def parse_attGAN_args(args_attack):
    with open(os.path.join('./AttGAN/output', args_attack.AttGAN.attgan_experiment_name, 'setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    args.test_int = args_attack.AttGAN.attgan_test_int
    args.num_test = args_attack.global_settings.num_test
    args.gpu = args_attack.global_settings.gpu
    args.load_epoch = args_attack.AttGAN.attgan_load_epoch
    args.multi_gpu = args_attack.AttGAN.attgan_multi_gpu
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)
    return args


"""Get Attribute Model Args"""
def parse(args=None):
    with open(os.path.join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

def ssim_loss(img1,img2):
    """input tensor, translate to np.array"""
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))
    ssim = structural_similarity(img1_np,img2_np,channel_axis=2,data_range=1)
    return ssim

def denorm(x):
    return (x+1)/2

def norm(x):
    return 2*x-1

"""Set Training Details"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 100
seed_code = 3407
epsilon = 0.05
num_attrs = 5 
val_imgs = 500
mse_loss = torch.nn.MSELoss().to(device)

"""Main Function"""
if __name__ == '__main__':
    

    set_seed(seed_code)
    config_parser = ArgumentParser()
    config_parser.add_argument('--gan_loss_type', default='lsgan', type=str, help='GAN Loss Type')
    config_parser.add_argument('--batch_size', default=4, type=int, help='Batch Size For Training')
    config_parser.add_argument('--lambda_visual', default=5, type=float, help='lambda_visual')
    config_parser.add_argument('--lambda_gan', default=0.01, type=float, help='lambda_gan')
    config_parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')


    """
    Parse Options Args
    """
    options = config_parser.parse_args()
    lambda_visual = options.lambda_visual 
    lambda_gan = options.lambda_gan
    batch_size = options.batch_size
    model_save_path = os.path.join('./sfgan_weights', 'attgan')
    os.makedirs(model_save_path, exist_ok=True)
    

    """
    Parse CelebA and attr Model
    """
    args_attack = parse()
    attgan_args = parse_attGAN_args(args_attack)
    train_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'train', attgan_args.attrs,args_attack.stargan.selected_attrs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,shuffle=True, drop_last=False)
    dataset_size = len(train_dataloader) 
    attgan_model = AttModel(attgan_args)
    attgan_model.eval().to(device)
    

    """Create sfgannet"""
    sfgannet = SFGANNet(options).to(device)
    parse_model=U2NetSODModel().to(device)

    for epoch in range(1, n_epochs):  # outer loop for different epochs;
        """Training Chapter Start"""
        train_current_loss = {'full_D': 0., 'fake_D': 0., 'real_D': 0.,'full_G': 0., 'adv': 0., 'visual': 0., 'gan': 0.}
        for idx, (img_a, att_a, c_org) in enumerate(tqdm(train_dataloader, desc='')):  # inner loop within one epoch
            img_a_cuda = img_a.to(device)
            att_a_cuda = att_a.cuda()
            att_a_cuda = att_a_cuda.type(torch.float)
            # Perform update and generate adversarial noise
            sod_face_mask = parse_model.sod_mask(img_a_cuda)
            """Get the adversarial noise by forward and Clamp the adversarial noise to within epsilon"""
            adv_noise = sfgannet.netG(img_a_cuda)*epsilon
            """Clamp the adv img_a to in RGB domain -1 1"""
            adv_img_a = torch.clamp(img_a_cuda+adv_noise,-1,1)
            
            
            """Forward It"""
            """Get the Original StarGAN result"""
            att_results_ori = attgan_model.att_modify(img_a_cuda,att_a_cuda)
            """Get the Adversarial StarGAN result"""
            att_results_adv = attgan_model.att_modify(adv_img_a,att_a_cuda)

        
            """Optimize Discriminator in sfgannet"""
            sfgannet.set_requires_grad(sfgannet.netD, True)
            sfgannet.optimizer_D.zero_grad()
            pred_adv_fake = sfgannet.netD(adv_img_a.detach())
            loss_D_adv_fake = sfgannet.gan_loss(pred_adv_fake, False)
            pred_real = sfgannet.netD(img_a_cuda)
            loss_D_real = sfgannet.gan_loss(pred_real, True)
            loss_D = (loss_D_adv_fake+loss_D_real)*0.5
            loss_D.backward()
            sfgannet.optimizer_D.step()


            """Optimize Generator in sfgannet"""
            sfgannet.set_requires_grad(sfgannet.netD, False)
            sfgannet.optimizer_G.zero_grad()
            pred_adv_real = sfgannet.netD(adv_img_a.detach())
            loss_gan = sfgannet.gan_loss(pred_adv_real, True)
            loss_adv = 0
            for i in range(num_attrs):
                loss_adv += mse_loss(att_results_ori[i]*sod_face_mask, att_results_adv[i]*sod_face_mask)
            loss_adv /= num_attrs
            loss_visual = mse_loss(adv_img_a,img_a_cuda)
            loss_G = (-1)*loss_adv + loss_visual*lambda_visual + loss_gan*lambda_gan
            loss_G.backward()
            sfgannet.optimizer_G.step()
   

            """Record Losses"""
            train_current_loss['full_D'] += loss_D
            train_current_loss['fake_D'] += loss_D_adv_fake
            train_current_loss['real_D'] += loss_D_real
            train_current_loss['full_G'] += loss_G
            train_current_loss['adv'] += loss_adv
            train_current_loss['visual'] += loss_visual
            train_current_loss['gan'] += loss_gan


        train_current_loss['full_D'] /= len(train_dataloader)
        train_current_loss['fake_D'] /= len(train_dataloader)
        train_current_loss['real_D'] /= len(train_dataloader)
        train_current_loss['full_G'] /= len(train_dataloader)
        train_current_loss['adv'] /= len(train_dataloader)
        train_current_loss['visual'] /= len(train_dataloader)
        train_current_loss['gan'] /= len(train_dataloader)
        print('epoch %d, loss D is %f, loss fake is %f, loss real is %f'%(epoch, train_current_loss['full_D'],train_current_loss['fake_D'],train_current_loss['real_D']))
        print('epoch %d, loss G is %f, loss adv is %f, loss visual is %.16f, loss gan is %f'%(epoch, train_current_loss['full_G'],train_current_loss['adv'],train_current_loss['visual'],train_current_loss['gan']))
        """Training Chapter End"""
        save_filename_model = 'sfgannet_sod_loss_attgan_%s_%s_epsilon_%s_epoch%s.pth'%(lambda_visual, lambda_gan, epsilon, epoch)
        save_path = os.path.join(model_save_path, save_filename_model)
        torch.save({"netG": sfgannet.netG.state_dict(), "netD": sfgannet.netD.state_dict()}, save_path)