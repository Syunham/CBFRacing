"""Physical consistency and CLI/CBF regressions for the integrated racing setup."""
import unittest
from unittest.mock import patch
from context import *
from validate import parse_args
from controller import RecoveryMPCC
from phase_planner import integrate_symbolic,GatePlanner

class RacingIntegration(unittest.TestCase):
    def test_defaults_and_explicit_branch_overrides(self):
        a=parse_args(['--name','unused'])
        self.assertEqual((a.cbf_branch,a.cbf_mode,a.frontend,a.preview_gates,a.progress,a.preview_spline,a.reference_refine),('learned','hard','phase',5,220.,'c2',2))
        self.assertTrue(a.physical_gate_guard and a.dense_physics and a.tube_recovery)
        a=parse_args(['--name','unused','--cbf','analytic','--frontend','original','--preview-gates','2','--progress','80','--no-physical-gate-guard','--iterations','17','--solver-seconds','3'])
        self.assertEqual((a.cbf_mode,a.frontend,a.preview_gates,a.progress,a.iterations,a.solver_seconds),('analytic','original',2,80.,17,3.))
        self.assertIsNone(a.h_checkpoint);self.assertFalse(a.physical_gate_guard or a.dense_physics)
        with self.assertRaises(SystemExit),patch('sys.stderr'):
            parse_args(['--name','unused','--frontend','original','--preview-gates','5'])

    def test_physical_units_and_dissipation(self):
        self.assertEqual((MODEL['mass'],MODEL['arm'],MODEL['thrust_max'],MODEL['omega_max']),(.85,.15,7.,10.))
        np.testing.assert_equal(MODEL['inertia'],[.0025,.0021,.0043]);np.testing.assert_equal(MODEL['drag'],[.26,.28,.42])
        rhs,_=model_functions(4,False);state=np.zeros(13);state[6]=1;state[3]=10;hover=np.full(4,MODEL['mass']*MODEL['gravity']/4)
        self.assertAlmostEqual(float(rhs(state,hover)[3]),-.26*10/.85,places=12)
        rng=np.random.default_rng(20261102);errors=[];powers=[];c=configuration();_,step=model_functions(32,False)
        for _ in range(30):
            x=np.zeros(13);x[3:6]=rng.normal(0,10,3);q=rng.normal(size=4);q/=np.linalg.norm(q);x[6:10]=q;x[10:13]=rng.normal(0,1,3);u=rng.uniform(0,7,4)
            frontend,_=integrate_symbolic(ca.DM(x),ca.DM(u),.08,c,32)
            errors.append(float(abs(np.array(frontend).ravel()-np.array(step(x,u,.08)).ravel()).max()))
            no_drag=dict(MODEL,drag=[0.,0.,0.]);without,_=model_functions(1,False,parameters=no_drag)
            dv=np.array(rhs(x,u)).ravel()[3:6]-np.array(without(x,u)).ravel()[3:6];powers.append(float(MODEL['mass']*dv@x[3:6]))
        self.assertLess(max(errors),1e-8);self.assertLessEqual(max(powers),0.)
        save_json(HERE/'checks/model_consistency_20261102.json',dict(passed=True,samples=30,maximum_difference=max(errors),drag_power_range=[min(powers),max(powers)]))

    def test_saved_model_replay_can_use_zero_drag_without_changing_live_model(self):
        state=np.zeros(13);state[6]=1;state[3]=10;u=np.full(4,MODEL['mass']*MODEL['gravity']/4)
        historical,_=model_functions(1,False,parameters=dict(MODEL,drag=[0.,0.,0.]));live,_=model_functions(1,False)
        self.assertEqual(float(historical(state,u)[3]),0.);self.assertLess(float(live(state,u)[3]),-3.)

    def test_both_safety_branches_keep_their_constraint_meaning(self):
        a=parse_args(['--name','unused']);n=3
        with patch.dict(os.environ,{'PHYSICAL_GATE_GUARD':'1'}):
            learned=RecoveryMPCC(0,n=n,cbf_checkpoint=a.h_checkpoint,cbf_mode='hard',dense_physics=True,tube_recovery=True)
            analytic=RecoveryMPCC(0,n=n,cbf_mode='analytic')
        self.assertEqual((len(learned.h_rows),len(learned.dh_rows)),(n+1,n));self.assertEqual(analytic.h_rows,[])
        self.assertEqual(learned.barrier_constraint_count,2*n+1);self.assertEqual(analytic.barrier_constraint_count,n+1)

    def test_gate_preview_wrap_and_phase_mesh_sizes(self):
        g,*_=gate_pair(6,5);np.testing.assert_equal(g,np.array(course()['centers'])[[6,0,1,2,3]])
        for count in (2,5):
            c=dict(configuration(),preview_gates=count,cells_per_leg=2);planner=GatePlanner(c,course(),True)
            self.assertEqual((planner.M,planner.n),(count,2*count))

if __name__=='__main__':unittest.main()
