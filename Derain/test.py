import torch
import torch.nn as nn


class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 2
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

layers = []
for _ in range(1):
    layers.append(Resblock())

f1 = nn.Sequential(*layers)
f2 = nn.Sequential(*layers)
print(f1)
# nn.init.constant_(f1.Sequential.conv1.bias, 0.0)
f1.state_dict()['0.conv1.bias'] = torch.tensor([-0.,  0.0397])
print(f1.state_dict()['0.conv1.bias'])
print(f2.state_dict()['0.conv1.bias'])

