import torch
from stargan.solver import Solver

class StarModel(torch.nn.Module):   #这个fsmodel继承了神经网络层的模块

    def __init__(self, dataloader, args_stargan):  #这里默认攻击的损失类型是L2范数
        super(StarModel, self).__init__() #这是调用了一个自己的初始化函数

        self.stargan_solver=Solver(celeba_loader=dataloader, rafd_loader=None, config=args_stargan)
        self.stargan_solver.restore_model(self.stargan_solver.test_iters)

    def star_modify(self,image,c_org):
        c_trg_list = self.stargan_solver.create_labels(c_org, self.stargan_solver.c_dim, self.stargan_solver.dataset, self.stargan_solver.selected_attrs)
        # Translated images.
        x_fake_list = []
        for idx, c_trg in enumerate(c_trg_list):
            # print('image', i, 'class', idx)
            x_real_mod = image
            # x_real_mod = self.blur_tensor(x_real_mod) # use blur
            gen_noattack,_ = self.stargan_solver.G(x_real_mod, c_trg)
            x_fake_list.append(gen_noattack)
        return x_fake_list