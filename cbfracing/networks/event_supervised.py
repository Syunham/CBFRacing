"""Pure supervised regression of five numerical-teacher flight durations."""
from cbfracing.core.context import *
from torch import nn

def event_features(state,gates,normals,half):
    state=np.array(state,dtype=float,copy=True);q=state[6:10]
    if q[0]<0:state[6:10]=-q
    return np.r_[state[:3]/10.,state[3:6]/15.,state[6:10],state[10:13]/10.,
                 ((np.asarray(gates)-state[:3])/15.).ravel(),np.asarray(normals).ravel(),(np.asarray(half)/.75).ravel()]

class EventNetwork(nn.Module):
    def __init__(self,input_dim=53,duration_min=.4):
        super().__init__();self.input_dim=input_dim;self.duration_min=duration_min;self.net=nn.Sequential(nn.Linear(input_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,5))
        self.register_buffer('xmean',torch.zeros(input_dim,dtype=torch.float64));self.register_buffer('xstd',torch.ones(input_dim,dtype=torch.float64))
        self.register_buffer('ymean',torch.zeros(5,dtype=torch.float64));self.register_buffer('ystd',torch.ones(5,dtype=torch.float64))
    def log_times(self,z):return self.ymean+self.ystd*self.net((z-self.xmean)/self.xstd)
    def forward(self,z):return torch.exp(self.log_times(z)).clamp(self.duration_min,8.)

class BoundedEventNetwork(nn.Module):
    """Learn bounded multiplicative corrections to a frozen time policy."""
    def __init__(self,input_dim=54,base_duration_min=.4,log_bound=.15):
        super().__init__();self.input_dim=input_dim;self.duration_min=.15;self.log_bound=log_bound
        self.base=EventNetwork(input_dim,base_duration_min)
        self.residual=nn.Sequential(nn.Linear(input_dim,64),nn.Tanh(),nn.Linear(64,5))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        for parameter in self.base.parameters():parameter.requires_grad_(False)
    def log_times(self,z):
        normalized=(z-self.base.xmean)/self.base.xstd
        return self.base(z).log()+self.log_bound*torch.tanh(self.residual(normalized))
    def forward(self,z):return self.log_times(z).exp().clamp(self.duration_min,8.)

class ShortSegmentEventNetwork(nn.Module):
    """Frozen five-event policy plus a learned bounded short-leg residual."""
    def __init__(self,input_dim=54,log_bound=.4):
        super().__init__();self.input_dim=input_dim;self.duration_min=.15;self.log_bound=log_bound
        self.base=BoundedEventNetwork(input_dim)
        self.short_head=nn.Sequential(nn.Linear(input_dim,32),nn.Tanh(),nn.Linear(32,5))
        nn.init.zeros_(self.short_head[-1].weight);nn.init.zeros_(self.short_head[-1].bias)
        for p in self.base.parameters():p.requires_grad_(False)
    def short_mask(self,z):
        relative=z[...,13:28].reshape(*z.shape[:-1],5,3)*15.
        differences=torch.cat((relative[...,:1,:],relative[...,1:,:]-relative[...,:-1,:]),dim=-2)
        return (torch.linalg.vector_norm(differences,dim=-1)<3.).to(z.dtype)
    def log_times(self,z):
        normalized=(z-self.base.base.xmean)/self.base.base.xstd
        return self.base(z).log()+self.short_mask(z)*self.log_bound*torch.tanh(self.short_head(normalized))
    def forward(self,z):return self.log_times(z).exp().clamp(self.duration_min,8.)

def load_event(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('architecture')=='short_segment_residual_v1':
        net=ShortSegmentEventNetwork(ck.get('input_dim',54),ck.get('log_bound',.4))
    elif ck.get('architecture')=='bounded_residual_v1':
        net=BoundedEventNetwork(ck.get('input_dim',54),ck.get('base_duration_min',.4),ck.get('log_bound',.15))
    else:
        net=EventNetwork(ck.get('input_dim',53),ck.get('duration_min',.4))
    net=net.double()
    net.load_state_dict(ck['state']);net.eval();return net,ck

def predict(net,state,gates,normals,half):
    features=extended_features(state,gates,normals,half,net.input_dim)
    with torch.no_grad():return net(torch.tensor(features[None],dtype=torch.float64))[0].numpy()

def extended_features(state,gates,normals,half,input_dim):
    features=event_features(state,gates,normals,half)
    if input_dim in (54,72):
        if len(state)<17:raise ValueError('This time network requires measured rotor thrusts in state[13:17]')
        features=np.r_[features,np.sum(state[13:17])/(4*MODEL['thrust_max'])]
    if input_dim==72:
        from scipy.spatial.transform import Rotation
        q=np.asarray(state[6:10]);R=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix();v=np.asarray(state[3:6]);n=np.asarray(normals)
        a=R[:,2]*sum(state[13:17])/MODEL['mass']-np.array([0,0,MODEL['gravity']])-R@np.diag(MODEL['drag'])@R.T@v/MODEL['mass']
        features=np.r_[features,a/35.,np.sum((np.asarray(gates)-state[:3])*n,axis=1)/15.,n@v/20.,n@a/35.]
    if len(features)!=input_dim:raise ValueError(f'Feature dimension{len(features)} differs from model{input_dim}')
    return features
