"""Frozen duration network used only to initialize free-time CPC."""
import torch,numpy as np
from torch import nn

class DurationNetwork(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.net=nn.Sequential(nn.Linear(26,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,2))
        nn.init.zeros_(self.net[-1].weight)
        frac=(np.asarray(c['initial_durations'])-c['duration_min'])/(c['duration_max']-c['duration_min'])
        with torch.no_grad(): self.net[-1].bias.copy_(torch.tensor(np.log(frac/(1-frac))))

    def forward(self,state,gates,normals,half):
        # Relative positions make absolute world translation irrelevant.
        feature=torch.cat([state[...,3:6]/5,state[...,6:10],state[...,10:13]/8,
            ((gates-state[...,None,:3])/3).flatten(-2),normals.flatten(-2),half.flatten(-2)],-1)
        return self.c['duration_min']+(self.c['duration_max']-self.c['duration_min'])*torch.sigmoid(self.net(feature))

class ArrivalTimeNetwork(torch.nn.Module):
    def __init__(self,checkpoint):
        super().__init__();self.duration_head=DurationNetwork(checkpoint['config']).double()
        self.duration_head.load_state_dict(checkpoint['policy'])
    def forward(self,state,gates,normals,half):
        return torch.cumsum(self.duration_head(state,gates,normals,half),dim=-1)
