import torch
from torch import nn

class Actor(nn.Module):
     
    def __init__(self, state_dim, n_actions, n_colors):
        super().__init__()
        self.state_nets = nn.ModuleList([nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, n_actions),
                        nn.Softmax(dim=-1)
        ) for i in range(n_colors)]) 

    def forward(self, X): 
        return self.state_nets[int(X[-1])](X)
        
    def save_model_weights(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path:str, device:str='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))

class Critic(nn.Module):

    def __init__(self, state_dim, n_colors):
        super().__init__()
        self.state_nets = nn.ModuleList([nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
        ) for i in range(n_colors)])

    def forward(self, X):
        return self.state_nets[int(X[-1])](X) 

    def save_model_weights(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path:str, device:str='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))