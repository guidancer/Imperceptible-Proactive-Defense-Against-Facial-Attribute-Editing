import torch
import argparse
from fgan.solver import Solver
def str2bool(v):
    return v.lower() in ('true')
def get_config():
    parser = argparse.ArgumentParser()
    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--crop_size', type=int, default=178, help='crop size for the images')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=10, help='weight for identity loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'BRATS', 'Directory'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_brats'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='celeba/logs')
    parser.add_argument('--model_save_dir', type=str, default='./fgan/ckpts')
    parser.add_argument('--sample_dir', type=str, default='celeba/samples')
    parser.add_argument('--result_dir', type=str, default='celeba/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    return config

class FModel(torch.nn.Module):   #这个fsmodel继承了神经网络层的模块

    def __init__(self, dataloader):  #这里默认攻击的损失类型是L2范数
        super(FModel, self).__init__() #这是调用了一个自己的初始化函数

        self.fgan_solver=Solver(data_loader=dataloader, config=get_config())
        self.fgan_solver.restore_model(self.fgan_solver.test_iters)

    def f_modify(self,image,c_org):
        c_trg_list = self.fgan_solver.create_labels(c_org, self.fgan_solver.c_dim, self.fgan_solver.dataset, self.fgan_solver.selected_attrs)
        # Translated images.
        x_fake_list = []
        for idx, c_trg in enumerate(c_trg_list):
            # print('image', i, 'class', idx)
            x_real_mod = image
            # x_real_mod = self.blur_tensor(x_real_mod) # use blur
            gen_noattack = torch.tanh(x_real_mod + self.fgan_solver.G(x_real_mod, c_trg))
            x_fake_list.append(gen_noattack)
        return x_fake_list