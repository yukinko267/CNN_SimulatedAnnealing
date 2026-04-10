import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # (in_channel, out_channel, kernel_size)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.fc = nn.Linear(12 * 4 * 4, 10)
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x)) # 6c, 28*28*1->24*24*6
        x = F.max_pool2d(x, 2) # 2s, 24*24*6->12*12*6
        x = torch.tanh(self.conv2(x)) # 12c, 12*12*6->8*8*12
        x = F.max_pool2d(x, 2) # 2s, 8*8*12->4*4*12

        x = x.view(x.size(0), -1) # convert 1 vecter
        x = self.fc(x) # classifier layer
        x = torch.softmax(x, dim=1) if CFG.SGD_or_SA == "SA" else x # only if SA

        return x
    

