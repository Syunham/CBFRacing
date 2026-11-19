"""Causal progress-cost scheduling from reference preview and solver feedback."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np
import json
from pathlib import Path
from scripts.control.rti_problem import DT


class ProgressReward:
    def __init__(self,args):
        self.args = args
        self.weight = args.progress_start if args.progress_start is not None else 220.
        self.last_time = None
        self.cooldown_until = -1.
        self.previous_residual = 0.
        self.previous_full_residual = 0.
        self.previous_near_residual = 0.
        self.residual_memory = 0.
        self.residual_observation_time = None
        self.residual_memory_tau = float(getattr(args,'adaptive_residual_memory_seconds',0.))
        if not np.isfinite(self.residual_memory_tau) or self.residual_memory_tau < 0:
            raise ValueError('Residual memory duration must be finite and nonnegative')
        self.residual_tail_weight=float(getattr(args,'adaptive_residual_tail_weight',1.))
        if not 0.<=self.residual_tail_weight<=1.:
            raise ValueError('Residual tail weight must be in [0,1]')
        self.diagnostics = {}
        self.gate_gains=np.ones(7)
        schedule=getattr(args,'flying_gate_schedule',None)
        if schedule:
            self.gate_gains=np.asarray(json.loads(Path(schedule).read_text(encoding='utf-8'))['gains'],dtype=float)
            if self.gate_gains.shape!=(7,) or not np.isfinite(self.gate_gains).all() or np.any((self.gate_gains<.9)|(self.gate_gains>1.1)):
                raise ValueError('Split-S flying-gate schedule requires seven gains in [.9,1.1]')

    def observe(self,result,time):
        self.previous_full_residual=max(0.,float(result.get('primal_residual',0.)))
        near=result.get('near_prediction_residual',self.previous_full_residual)
        if result.get('nlp_recovery'):near=self.previous_full_residual
        self.previous_near_residual=max(0.,float(near))
        self.previous_residual=max(self.previous_near_residual,self.residual_tail_weight*self.previous_full_residual)
        if self.residual_memory_tau > 0:
            elapsed=0. if self.residual_observation_time is None else max(0.,time-self.residual_observation_time)
            self.residual_memory=max(self.previous_residual,self.residual_memory*np.exp(-elapsed/self.residual_memory_tau))
        self.residual_observation_time=time
        failed_qp = any(not row['qp'].get('success',True)
                        for row in result.get('solve_trace',[]) if isinstance(row.get('qp'),dict))
        if result['status'] or result.get('sqp_recovery') or result.get('progress_backoff') or failed_qp:
            self.cooldown_until = time+.5

    def update(self,time,state,reference,solver,completed_laps=0,gate_index=0):
        if self.last_time == time:
            return self.weight
        args = self.args
        progress = float(np.clip(state[17],0.,reference['length']))
        query = np.linspace(progress,min(reference['length'],progress+25.),101)
        d1,d2 = reference['spline'](query,1),reference['spline'](query,2)
        curvature = np.linalg.norm(np.cross(d1,d2),axis=1)/np.maximum(np.linalg.norm(d1,axis=1)**3,1e-8)
        speed = np.interp(query,reference['speed_arc'],reference['speed_values'])+args.reward_speed_allowance
        curve_speed = np.sqrt(args.adaptive_lateral_acceleration/np.maximum(curvature,1e-4))
        desired = float(np.sqrt(np.min(np.minimum(speed,curve_speed)**2+2*args.reward_braking*(query-progress))))
        turn_preview=getattr(args,'adaptive_turn_preview',8.)
        turn = float(np.clip((curvature[query-progress<=turn_preview].max()-.05)/.15,0.,1.))
        turn_cap = args.turn_reward if args.turn_reward>0 else args.progress
        ceiling = (1-turn)*args.progress+turn*min(args.progress,turn_cap)
        flying_gain=getattr(args,'adaptive_flying_gain',1.) if completed_laps>=1 else 1.
        ceiling*=flying_gain
        gate_gain=float(self.gate_gains[gate_index%7]) if completed_laps>=1 else 1.
        ceiling*=gate_gain
        error = (desired-np.linalg.norm(state[3:6]))/args.reward_transition
        speed_factor = float(1/(1+np.exp(-np.clip(error,-30.,30.))))
        tangent = d1[0]/max(np.linalg.norm(d1[0]),1e-8)
        position_error = state[:3]-reference['spline'](progress)
        contour = float(np.linalg.norm(position_error-tangent*(position_error@tangent)))
        tracking_factor = max(.25,1/(1+(contour/args.adaptive_tracking_scale)**2))
        minimum_h = None;minimum_decay_headroom=None;decay_factor=1.
        safety_factor = 1.
        # Uniform ablation: no barrier-dependent extra reward safety mechanism.
        if False:
            X,_ = solver.prediction()
            count = min(4,len(X))
            hs = [float(solver.neural(X[k],solver.od[k,16:22])) for k in range(count)]
            minimum_h = min(hs)
            safety_factor = float(np.clip(minimum_h/args.adaptive_safety_h,.2,1.))
            reserve=getattr(args,'adaptive_decay_reserve',0.)
            if reserve>0 and count>1:
                hs[0]=float(solver.neural(state,solver.od[0,16:22]))
                headroom=np.asarray(hs[1:])-np.exp(-solver.cbf_alpha*DT)*np.asarray(hs[:-1])-args.cbf_decay_margin*DT
                minimum_decay_headroom=float(headroom.min())
                decay_factor=float(np.clip(minimum_decay_headroom/reserve,.2,1.))
        recovery_factor = .35 if time<self.cooldown_until else 1.
        residual_factor = 1.
        tail_weight=self.residual_tail_weight
        if getattr(args,'adaptive_residual_straights_only',False):
            tail_weight=1.-(1.-tail_weight)*(1.-turn)
        self.previous_residual=max(self.previous_near_residual,tail_weight*self.previous_full_residual)
        if self.residual_memory_tau > 0:
            self.previous_residual=max(self.previous_residual,self.residual_memory)
        if args.adaptive_solve_tolerance>0:
            residual_factor = float(np.clip(args.adaptive_solve_tolerance/max(self.previous_residual,1e-12),.15,1.))
        target = args.reward_floor+(ceiling-args.reward_floor)*speed_factor*tracking_factor*safety_factor*recovery_factor*residual_factor*decay_factor
        pre_ramp_target=target
        if args.progress_start is not None:
            fraction = float(np.clip((time-args.progress_ramp_delay)/args.progress_ramp_seconds,0.,1.))
            target = args.progress_start+(target-args.progress_start)*fraction
            if getattr(args,'progress_ramp_increase_only',False):target=min(pre_ramp_target,target)
        dt = .02 if self.last_time is None else max(0.,time-self.last_time)
        self.weight += float(np.clip(target-self.weight,-args.adaptive_fall_rate*dt,args.adaptive_rise_rate*dt))
        self.last_time = time
        self.diagnostics = dict(weight=self.weight,target=target,desired_speed=desired,
            pre_ramp_target=pre_ramp_target,ramp_increase_only=bool(getattr(args,'progress_ramp_increase_only',False)),
            maximum_preview_curvature=float(curvature.max()),contour_error=contour,
            speed_factor=speed_factor,tracking_factor=tracking_factor,minimum_near_predicted_h=minimum_h,
            safety_factor=safety_factor,recovery_factor=recovery_factor,
            previous_prediction_residual=self.previous_residual,residual_factor=residual_factor,
            previous_full_prediction_residual=self.previous_full_residual,
            previous_near_prediction_residual=self.previous_near_residual,residual_tail_weight=tail_weight,
            residual_memory=self.residual_memory,residual_memory_seconds=self.residual_memory_tau,
            minimum_decay_headroom=minimum_decay_headroom,decay_factor=decay_factor,
            turn_preview=turn_preview,turn_factor=turn,reward_ceiling=ceiling,
            flying_gain=flying_gain,gate_gain=gate_gain,gate_index=int(gate_index),completed_laps=int(completed_laps))
        return self.weight
