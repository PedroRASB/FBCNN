import torch
import torch.nn as nn

class SVM(nn.Module):

    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(45, len(targets))  # Implement the Linear function
        
    def forward(self, x):
        #x = x.view(-1, 20 * 5)
        fwd = self.fully_connected(x)  # Forward pass
        return fwd


    