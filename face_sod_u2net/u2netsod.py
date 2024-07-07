import os
import torch
from face_sod_u2net.model import U2NET
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class U2NetSODModel(torch.nn.Module):
    def __init__(self):
        super(U2NetSODModel,self).__init__()
        model_dir = os.path.join('./face_sod_u2net','ckpts','u2net_bce_itr_22500_train_0.138353_tar_0.013147.pth')
        self.sod_net = U2NET(3,1)
        self.sod_net.load_state_dict(torch.load(model_dir))
        self.sod_net.eval().to(device)

    def sod_mask(self,image):
        return self.sod_net(image)