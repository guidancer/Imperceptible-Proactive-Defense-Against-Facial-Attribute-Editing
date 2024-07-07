import os
import torch
from face_sod_u2net.model import U2NETP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class U2NetPSODModel(torch.nn.Module):
    def __init__(self):
        super(U2NetPSODModel,self).__init__()
        model_dir = os.path.join('./face_sod_u2net','ckpts','u2netp_bce_itr_16500_train_0.201991_tar_0.020847.pth')
        self.sod_net = U2NETP(3,1)
        self.sod_net.load_state_dict(torch.load(model_dir))
        self.sod_net.eval().to(device)

    def sod_mask(self,image):
        return self.sod_net(image)