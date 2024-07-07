import torch
from AttentionGAN.AttentionGAN_v1_multi.solver import Solver

class AttentionModel(torch.nn.Module):   #这个fsmodel继承了神经网络层的模块

    def __init__(self, dataloader, args_attentiongan):  #这里默认攻击的损失类型是L2范数
        super(AttentionModel, self).__init__() #这是调用了一个自己的初始化函数

        self.attentiongan_solver=Solver(celeba_loader=dataloader, rafd_loader=None, config=args_attentiongan)
        self.attentiongan_solver.restore_model(self.attentiongan_solver.test_iters)

    def attention_modify(self,image,c_org):
        c_trg_list = self.attentiongan_solver.create_labels(c_org, self.attentiongan_solver.c_dim, self.attentiongan_solver.dataset, self.attentiongan_solver.selected_attrs)
        x_fake_list=[]
        # Set data loader.
        for i, c_trg in enumerate(c_trg_list):
            # Prepare input images and target domain labels.
            # x_fake_list.append(self.G(x_real, c_trg))
            gen_noattack, attention, _ = self.attentiongan_solver.G(image, c_trg)
            attention = (attention - 0.5) / 0.5
            x_fake_list.append(gen_noattack)
            # Attacks
        return x_fake_list