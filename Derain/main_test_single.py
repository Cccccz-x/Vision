import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from model import DeRain
import h5py
import scipy.io as sio
import os
from data import TestDatasetMat
from torch.utils.data import DataLoader


###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = "Weights/400.pth"   # chose model

def test(file_path):

    dataset = TestDatasetMat(file_path)
    dataloader = DataLoader(dataset, 3)
    model = DeRain().eval()   # fixed, important!
    # weight = torch.load(ckpt)  # load Weights!
    # model.load_state_dict(weight) # fixed
    
    with torch.no_grad():
        for batch in dataloader:
            img, gt = batch['img'], batch['gt']

            img = img.cpu().permute(0, 3, 1, 2).float()  
            # gt = gt.cpu().permute(0, 3, 1, 2).float()
            print(img.shape)

            output = model(img)
            
            output = output.permute(0, 2, 3, 1).cpu().detach().numpy()  # HxWxC

        print(output.shape)
        save_name = os.path.join("test_results", "derain.mat")  # fixed! save as .mat format that will used in Matlab!
        sio.savemat(save_name, {'derain': output})  # fixed!

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "derain_small/test12_chunks.mat"
    test(file_path)   # recall test function
