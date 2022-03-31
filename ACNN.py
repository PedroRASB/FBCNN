import torch
import torch.nn as nn

class SmallNet(nn.Module):
    #ACNN
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1= nn.Conv2d(1, 16, kernel_size=(6, 6), stride=(1, 1), padding=(1, 1))
        self.bn1=nn.BatchNorm2d(16)
        self.relu1= nn.ReLU()
        self.pool1=torch.nn.MaxPool2d(kernel_size=(2, 3))
        self.drop1= nn.Dropout(p=0.25) 
        
        self.conv2= nn.Conv2d(16, 32, kernel_size=(10, 1), stride=(1, 1), padding=(1, 0))
        self.bn2=nn.BatchNorm2d(32)
        self.relu2= nn.ReLU()
        self.pool2=torch.nn.MaxPool2d(kernel_size=(2, 1))
        self.drop2= nn.Dropout(p=0.25) 
        
        self.outL= nn.Linear(7*32,len(targets))
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.drop1(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)

        x=x.view(x.size(0), -1)
        
        x=self.drop2(x)
        x=self.outL(x)
        return(x)


    