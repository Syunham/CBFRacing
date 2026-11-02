"""Direct scalar neural h, without analytic residual or learned margin."""
from context import *
from torch import nn

def features(x,y):  # input features
    x=np.asarray(x);y=np.asarray(y);r=x[...,:3]-y[...,:3];v=x[...,3:6]-y[...,3:6]
    d=np.sqrt(np.sum(r*r,axis=-1,keepdims=True)+1e-10);closing=np.sum(r*v,axis=-1,keepdims=True)/d
    return np.concatenate([r/3.,v/12.,x[...,3:6]/12.,x[...,6:10],x[...,10:13]/10.,x[...,13:17]/7.,d/3.,closing/12.],axis=-1)

class DirectCBF(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width
        self.net=nn.Sequential(nn.Linear(22,width),nn.Tanh(),nn.Linear(width,16),nn.Tanh(),nn.Linear(16,1),nn.Tanh())
    def forward(self,z):return self.net(z).squeeze(-1)

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);net=DirectCBF(ck['width']).double();net.load_state_dict(ck['state']);net.eval();return net,ck

def symbolic(net):  # torch->casadi
    x=ca.MX.sym('own',19);y=ca.MX.sym('other',6);r=x[:3]-y[:3];v=x[3:6]-y[3:6]
    d=ca.sqrt(ca.sumsqr(r)+1e-10)
    z=ca.vertcat(r/3.,v/12.,x[3:6]/12.,x[6:10],x[10:13]/10.,x[13:17]/7.,d/3.,ca.dot(r,v)/d/12.)
    for layer in net.net:
        if isinstance(layer,nn.Linear):z=ca.DM(layer.weight.detach().numpy())@z+ca.DM(layer.bias.detach().numpy())
        else:z=ca.tanh(z)
    return ca.Function('direct_h',[x,y],[z])

def check(net,output):
    fn=symbolic(net);sx=ca.MX.sym('s',19);sy=ca.MX.sym('y',6);v=fn(sx,sy);jac=ca.Function('direct_jac',[sx,sy],[ca.jacobian(v,ca.vertcat(sx,sy))])
    rng=np.random.default_rng(141);maximum=0.;gradient=0.
    for _ in range(12):
        x=np.zeros(19);x[:6]=rng.normal(0,2,6);x[6:10]=rng.normal(size=4);x[6:10]/=np.linalg.norm(x[6:10]);x[10:13]=rng.normal(size=3);x[13:17]=rng.uniform(0,7,4);y=rng.normal(0,2,6)
        with torch.no_grad():h=float(net(torch.tensor(features(x,y)[None])))
        maximum=max(maximum,abs(h-float(fn(x,y))));analytic=np.asarray(jac(x,y)).ravel();state=np.r_[x,y];numeric=[]
        for j in range(25):
            delta=np.zeros(25);delta[j]=1e-6;a=state+delta;b=state-delta
            numeric.append((float(fn(a[:19],a[19:]))-float(fn(b[:19],b[19:])))/2e-6)
        gradient=max(gradient,float(abs(analytic-numeric).max()))
    assert maximum<1e-10 and gradient<2e-5
    save_json(output,dict(torch_casadi_error=maximum,gradient_error=gradient,states=12,gradient_dimensions=25,output='Direct tanh MLP scalar, no analytical output residual'))
