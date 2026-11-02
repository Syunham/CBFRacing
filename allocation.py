"""Random-initialized allocation policy. No teacher or label-generation API."""
from context import *
from torch import nn

def observation(own,other,gate,normal,axis,next_gate,next_normal):
    r=own[:3]-other[:3];v=own[3:6]-other[3:6]
    rel=np.r_[np.array([r@axis,r@normal,r[2]])/2.,np.array([v@axis,v@normal,v[2]])/8.]
    common=np.r_[(gate-(own[:3]+other[:3])/2)/10.,(own[3:6]+other[3:6])/16.,normal,axis,(next_gate-gate)/15.,next_normal]
    return np.r_[rel,common]

class GateAllocator(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(24,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,1))
    def forward(self,z):
        reverse=torch.cat([-z[:,:6],z[:,6:]],dim=1)
        return .45*torch.tanh((self.net(z)-self.net(reverse)).squeeze(-1))

def load_allocator(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);net=GateAllocator().double();net.load_state_dict(ck['state']);net.eval();return net

def save_allocator(path,net,**metadata):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=net.state_dict(),initialization='random; no supervised allocation weights loaded',teacher_labels_used=False,**metadata),path)
