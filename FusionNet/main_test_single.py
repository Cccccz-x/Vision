# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from model import FusionNet
import h5py
import scipy.io as sio
import os

###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################
# def get_edge(data):  # get high-frequency
#     rs = np.zeros_like(data)
#     if len(rs.shape) == 3:
#         for i in range(data.shape[2]):
#             rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
#     else:
#         rs = data - cv2.boxFilter(data, -1, (5, 5))
#     return rs

def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047.0)  # HxW = 256x256

    return lms, ms, pan

###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = "Weights/400.pth"   # chose model

def test(file_path):
    lms, ms, pan = load_set(file_path)

    model = FusionNet().eval()   # fixed, important!
    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    with torch.no_grad():

        lms = lms.cpu().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        pan = pan.cpu().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
        print(lms.shape, pan.shape)
        sr = model(lms, pan)
        sr = lms + sr   
        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC

        print(sr.shape)
        save_name = os.path.join("test_results", "new_data6_fusionnet.mat")  # fixed! save as .mat format that will used in Matlab!
        sio.savemat(save_name, {'new_data6_fusionnet': sr})  # fixed!

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "test_data/new_data6.mat"
    test(file_path)   # recall test function
