import torch
import os
from AttGAN.attgan import AttGAN
from AttGAN.utils import find_model
from AttGAN.data import check_attribute_conflict

class AttModel(torch.nn.Module):   #这个fsmodel继承了神经网络层的模块

    def __init__(self, args_attgan):  #这里默认攻击的损失类型是L2范数
        super(AttModel, self).__init__() #这是调用了一个自己的初始化函数

        self.attgan = AttGAN(args_attgan)
        self.attgan.load(find_model(os.path.join('./AttGAN/output', args_attgan.experiment_name, 'checkpoint'), args_attgan.load_epoch))
        self.attgan.eval()
        self.attgan_args = args_attgan
        self.target_attr = (3, 4, 5, 8, 13)

    def att_modify(self,img_a,att_a):
        x_fake_list = []
        att_b_list = [att_a]
        for i in range(self.attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, self.attgan_args.attrs[i], self.attgan_args.attrs)
            att_b_list.append(tmp)

        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
            if i > 0 :
                att_b_[..., i - 1] = att_b_[..., i - 1] * self.attgan_args.test_int / self.attgan_args.thres_int
                if i in self.target_attr:
                    gen_noattack = self.attgan.G(img_a, att_b_)
                    x_fake_list.append(gen_noattack)
        
        return x_fake_list
    
    