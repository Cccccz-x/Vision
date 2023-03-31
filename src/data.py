import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np


# def get_edge(data):  # input的维度排列顺序: HxWxC  =》pytorch要求的维度排列顺序: CxHxW
#     rs = np.zeros_like(data)
#     N = data.shape[0]
#     for i in range(N):
#         if len(data.shape) == 3:
#             rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
#         else:
#             rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
#     return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=Nx8x64x64
        # print(data)
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.
        self.gt = torch.from_numpy(gt1)  # NxCxHxW: 8x64x64

        print(self.gt.size())

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.
        self.lms = torch.from_numpy(lms1) # 8x64x64

        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3   8x16x16
        ms1 = np.array(ms1, dtype=np.float32) / 2047.  # NxHxWxC
        self.ms_hp = torch.from_numpy(ms1) # NxCxHxW:

        pan1 = data['pan'][...]  # Nx1xHxW  1x64x64
        pan1 = np.array(pan1, dtype=np.float32) / 2047.  # NxHxWx1
        self.pan_hp = torch.from_numpy(pan1) # Nx1xHxW:

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float(), \
               self.pan_hp[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

# validate_set = Dataset_Pro('training_data/valid_small.h5')
