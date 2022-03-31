import torch
import torch.nn as nn
import numpy as np
import scipy

def Filterbank(x, sampling, filterIdx):  
    #x: time signal, np array with format (electrodes,data)
    #sampling: sampling frequency
    #filterIdx: filter index

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Nq = sampling/2
    Wp = [passband[filterIdx]/Nq, 90/Nq]
    Ws = [stopband[filterIdx]/Nq, 100/Nq]
    [N, Wn] = scipy.signal.cheb1ord(Wp, Ws, 3, 40)
    [B, A] = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass')
    y = np.zeros(x.shape)
    channels = x.shape[0]
    for c in range(channels):
        y[c, :] = scipy.signal.filtfilt(B, A, x[c, :], padtype = 'odd', padlen=3*(max(len(B),len(A))-1), axis=-1)
    return y

    
class SmallNet(nn.Module):
    #FBCNN3D
    def __init__(self):
        super(FBCNN3D, self).__init__()
        self.conv1= nn.Conv3d(1, 16, kernel_size=(4, 6, 6), stride=(1, 1,1), padding=(1, 1,1))
        self.bn1=nn.BatchNorm3d(16)
        self.relu1= nn.LeakyReLU()
        self.pool1=torch.nn.MaxPool3d(kernel_size=(2, 2, 3))
        self.drop1= nn.Dropout(p=0.25) 
        
        self.conv2= nn.Conv3d(16, 32, kernel_size=(4, 10, 1), stride=(1, 1,1), padding=(1, 1,0))
        self.bn2=nn.BatchNorm3d(32)
        self.relu2= nn.LeakyReLU()
        self.pool2=torch.nn.MaxPool3d(kernel_size=(3, 2, 1))
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
    
