"""Causal velocity/acceleration forecasts; no shared plans or future observations."""
import numpy as np

def decaying_acceleration(position,velocity,acceleration,times,tau=.35):
    t=np.asarray(times)[:,None];a=np.asarray(acceleration)[None,:];v=np.asarray(velocity)[None,:]
    e=np.exp(-t/tau)
    return position+v*t+a*tau*(t-tau*(1-e)),v+a*tau*(1-e),a*e

def coordinated_turn(position,velocity,acceleration,times,tau=.35):
    t=np.asarray(times);v=np.asarray(velocity);a=np.asarray(acceleration);speed=np.linalg.norm(v[:2])
    if speed<1.:return decaying_acceleration(position,v,a,t,tau)
    direction=(v[0]+1j*v[1])/speed
    rate=np.clip((v[0]*a[1]-v[1]*a[0])/speed**2,-4.,4.)
    longitudinal=max(float(v[:2]@a[:2]/speed),-speed/tau)
    def integral(z):
        return t.astype(complex) if abs(z)<1e-10 else np.expm1(z*t)/z
    rotation=np.exp(1j*rate*t);tail=np.exp(-t/tau)
    distance=direction*((speed+longitudinal*tau)*integral(1j*rate)-longitudinal*tau*integral(-1/tau+1j*rate))
    xy_velocity=direction*rotation*(speed+longitudinal*tau*(1-tail))
    xy_acceleration=direction*rotation*(longitudinal*tail+1j*rate*(speed+longitudinal*tau*(1-tail)))
    p,w,b=decaying_acceleration(position,v,a,t,tau)
    p[:,:2]=position[:2]+np.c_[distance.real,distance.imag]
    w[:,:2]=np.c_[xy_velocity.real,xy_velocity.imag];b[:,:2]=np.c_[xy_acceleration.real,xy_acceleration.imag]
    return p,w,b

def turn_tail(position,velocity,acceleration,times,tau=.35):
    t=np.asarray(times);base=decaying_acceleration(position,velocity,acceleration,t,tau);turn=coordinated_turn(position,velocity,acceleration,t,tau)
    span=.4;s=np.clip((t-.24)/span,0.,1.)[:,None]
    w=10*s**3-15*s**4+6*s**5;dw=30*s**2*(1-s)**2/span;ddw=60*s*(1-s)*(1-2*s)/span**2
    dp=turn[0]-base[0];dv=turn[1]-base[1]
    return base[0]+w*dp,base[1]+w*dv+dw*dp,base[2]+w*(turn[2]-base[2])+2*dw*dv+ddw*dp
