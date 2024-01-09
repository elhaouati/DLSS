from torch import nn
import torch

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lr, hr_seg, hr_depth): 
        x = torch.cat((lr, hr_seg, hr_depth), dim=1)  
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x)) 
        x = self.conv3(x) 
        return x

    def forward_old(self, x):
        # Ajouter une dimension de lot au début
        x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # Ajouter une dimension de lot au début
        #x = x.unsqueeze(1)
        
        return x
