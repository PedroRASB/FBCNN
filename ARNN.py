import torch
import torch.nn as nn

class FBRNN(nn.Module):
    #Simple A-RNN
    def __init__(self):
        super(FBRNN, self).__init__()
        drop=0.4
        self.conv1= nn.Conv1d(1, 8, kernel_size=(32))
        self.bn1=nn.BatchNorm1d(8)
        self.relu1= nn.ReLU()
        self.pool1= nn.MaxPool1d(kernel_size=(2), stride=(2))
        self.drop1= nn.Dropout(p=drop)
        
        self.conv2= nn.Conv1d(8, 10, kernel_size=(32))
        self.bn2=nn.BatchNorm1d(10)
        self.relu2= nn.ReLU()
        self.pool2= nn.MaxPool1d(kernel_size=(2), stride=(2))
        self.drop2= nn.Dropout(p=drop)
        
        self.LSTM1= nn.LSTM(input_size=10,hidden_size=100,num_layers=1,batch_first=True,bidirectional=False)
        self.drop3= nn.Dropout(p=drop)
        self.LSTM2= nn.LSTM(input_size=100,hidden_size=50,num_layers=1,batch_first=True,bidirectional=False)
        self.drop4= nn.Dropout(p=drop)
        self.LSTM3= nn.LSTM(input_size=50,hidden_size=20,num_layers=1,batch_first=True,bidirectional=False)
        self.drop5= nn.Dropout(p=drop)
        self.LSTM4= nn.LSTM(input_size=20,hidden_size=10,num_layers=1,batch_first=True,bidirectional=False)
        self.drop6= nn.Dropout(p=drop)
        self.LSTM5= nn.LSTM(input_size=10,hidden_size=5,num_layers=1,batch_first=True,bidirectional=False)
        self.drop7= nn.Dropout(p=drop) 
        
        self.outL= nn.Linear(40,2)
        
    def forward(self,x):
        x=self.conv1(x)
        #print(x.shape)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.drop1(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=self.drop2(x)
        x=x.permute(0,2,1)#batch,sequence,channels
        #print(x.shape)
        
        x,_=self.LSTM1(x)
        x=self.drop3(x)
        x,_=self.LSTM2(x)
        x=self.drop4(x)
        x,_=self.LSTM3(x)
        x=self.drop5(x)
        x,_=self.LSTM4(x)
        x=self.drop6(x)
        x,_=self.LSTM5(x)
        x=self.drop7(x)
        #print(x.shape)
        
        x=x.reshape(x.size(0), -1)
        #print(x.shape)
        x=self.outL(x)
        
        return(x)


    