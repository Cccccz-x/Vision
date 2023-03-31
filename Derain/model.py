import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int

# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

# -----------------------------------------------------
class DeRain(nn.Module):
    def __init__(self):
        super(DeRain, self).__init__()

        channel = 32
        spectral_num = 3


        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU = nn.LeakyReLU()

        layers = []
        for _ in range(8):
            layers.append(Resblock())

        self.f1 = nn.Sequential(
            nn.Sequential(*layers)
        )
        
        self.f2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Sequential(*layers),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.f3 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Sequential(*layers),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )

        self.conv2 = nn.Conv2d(in_channels=channel * 3, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1)

        init_weights(self.f1, self.f2, self.f3, self.conv1, self.conv2, self.conv3)   # state initialization, important!

    def forward(self, x):  # x= hp of ms; # Bsx8x64x64 y = hp of pan # Bsx1x64x64
        x0 = self.LeakyReLU(self.conv1(x))
        x1 = self.f1(x0)
        x2 = self.f2(x0)
        x3 = self.f3(x0)
        y = torch.concat([x1, x2, x3], 1)
        y = self.conv2(y)
        y = self.conv3(y)
        return x - y



# ----------------- End-Main-Part ------------------------------------

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d

                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor


def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(3, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model,(x,))



def inspect_weight_decay():
    ...


net = DeRain()