"""Geometric initial guess for the numerical trajectory solver."""
import torch,math

def seed_states(initial,positions,config):
    """Geometric initialization; repaired when the quaternion conversion is singular."""
    h=config['dt'];q,nodes,_=positions.shape
    final_velocity=(torch.zeros_like(initial[:,None,3:6]) if config.get('terminal_rest',True)
                    else (positions[:,-1:]-positions[:,-2:-1])/h)
    v=torch.cat([initial[:,None,3:6],(positions[:,2:]-positions[:,:-2])/(2*h),
                 final_velocity],dim=1)
    acceleration=(positions[:,2:]-2*positions[:,1:-1]+positions[:,:-2])/h**2
    acceleration=torch.cat([torch.zeros_like(positions[:,:1]),acceleration,torch.zeros_like(positions[:,:1])],dim=1)
    force=acceleration+positions.new_tensor([0,0,config['vehicle']['gravity']])
    zb=force/force.norm(dim=-1,keepdim=True)
    heading=positions.new_tensor([1,0,0]).expand_as(zb)
    yb=torch.linalg.cross(zb,heading,dim=-1);yb=yb/yb.norm(dim=-1,keepdim=True)
    xb=torch.linalg.cross(yb,zb,dim=-1);rotation=torch.stack([xb,yb,zb],dim=-1)
    qw=.5*(1+rotation.diagonal(dim1=-2,dim2=-1).sum(-1)).sqrt()
    xyz=torch.stack([rotation[...,2,1]-rotation[...,1,2],rotation[...,0,2]-rotation[...,2,0],
        rotation[...,1,0]-rotation[...,0,1]],dim=-1)/(4*qw[...,None])
    quat=torch.cat([qw[...,None],xyz],dim=-1)
    qdot=(quat[:,2:]-quat[:,:-2])/(2*h)
    qdot=torch.cat([torch.zeros_like(quat[:,:1]),qdot,torch.zeros_like(quat[:,:1])],dim=1)
    omega=2*(quat[...,:1]*qdot[...,1:]-qdot[...,:1]*quat[...,1:]-torch.linalg.cross(quat[...,1:],qdot[...,1:],dim=-1))
    x=torch.cat([positions,v,quat,omega],-1)
    x=torch.cat([initial[:,None],x[:,1:]],dim=1)
    angular_acc=(omega[:,1:]-omega[:,:-1])/h
    midpoint_omega=(omega[:,1:]+omega[:,:-1])/2;vehicle=config['vehicle'];inertia=positions.new_tensor(vehicle['inertia'])
    tau=angular_acc*inertia+torch.linalg.cross(midpoint_omega,midpoint_omega*inertia,dim=-1)
    collective=vehicle['mass']*(force[:,1:].norm(dim=-1)+force[:,:-1].norm(dim=-1))/2
    arm=vehicle['arm_length']/math.sqrt(2);torque=vehicle['torque_coefficient']
    allocation=positions.new_tensor([[1,1,1,1],[arm,arm,-arm,-arm],[-arm,arm,arm,-arm],[torque,-torque,torque,-torque]])
    u=torch.cat([collective[...,None],tau],dim=-1)@torch.linalg.inv(allocation).T
    u=u.clamp(vehicle['thrust_min'],vehicle['thrust_max'])
    return x.transpose(1,2),u.transpose(1,2)
