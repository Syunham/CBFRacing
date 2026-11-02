"""Regression checks for the original analytic NLP and the learned-CBF entry."""
import unittest
from context import *
from validate import parse_args
from controller import RecoveryMPCC

class SafetyBranches(unittest.TestCase):
    def test_original_analytic_nlp_is_preserved(self):
        data=np.load(HERE/'checks/analytic_nlp_before_direct.npz');solver=RecoveryMPCC(0,n=int(data['n']))
        for key in ['lbg','ubg','lbx','ubx']:np.testing.assert_array_equal(getattr(solver,key),data[key])
        f=solver.solver.get_function('nlp_f');g=solver.solver.get_function('nlp_g')
        for k,(x,p) in enumerate(zip(data['x'],data['p'])):
            np.testing.assert_allclose(float(f(x,p)),data['f'][k],rtol=1e-13,atol=1e-9)
            np.testing.assert_allclose(np.asarray(g(x,p)).ravel(),data['g'][k],rtol=1e-13,atol=1e-9)
        args=parse_args(['--name','unused'])
        self.assertEqual((args.cbf_branch,args.cbf_mode,args.iterations,args.solver_seconds,args.gate_halfwidth,args.phase_retries),('analytic','analytic',900,8.,.20,0))
        self.assertIsNone(args.h_checkpoint);self.assertFalse(args.dense_physics or args.tube_recovery or args.retry_infeasible)

    def test_learned_profile_uses_direct_h_without_geometric_rows(self):
        args=parse_args(['--name','unused','--cbf','learned'])
        self.assertEqual((args.cbf_mode,args.iterations,args.solver_seconds,args.gate_halfwidth,args.phase_retries),('hard',4000,35.,.16,2))
        self.assertTrue(args.dense_physics and args.tube_recovery and args.retry_infeasible)
        provenance=json.loads((HERE/'PROVENANCE.json').read_text())
        self.assertEqual(hashlib.sha256(Path(args.h_checkpoint).read_bytes()).hexdigest(),provenance['direct_cbf_sha256'])
        n=3;solver=RecoveryMPCC(0,n=n,cbf_checkpoint=args.h_checkpoint,cbf_mode=args.cbf_mode,cbf_alpha=args.cbf_alpha,dense_physics=args.dense_physics,tube_recovery=args.tube_recovery)
        self.assertEqual((len(solver.h_rows),len(solver.dh_rows)),(n+1,n))
        # Replace n+1 analytic-distance rows by neural h and decay rows; remove
        # the two current funnel rows and add sixteen dense physical rows.
        old=np.load(HERE/'checks/analytic_nlp_before_direct.npz')
        self.assertEqual(solver.constraint_count,len(old['lbg'])-(n+1)+(2*n+1)-2+16)

    def test_explicit_overrides_survive_branch_defaults(self):
        args=parse_args(['--name','unused','--cbf','learned','--iterations','17','--solver-seconds','3','--gate-halfwidth','.18','--no-dense-physics','--phase-retries','1'])
        self.assertEqual((args.iterations,args.solver_seconds,args.gate_halfwidth,args.phase_retries),(17,3.,.18,1));self.assertFalse(args.dense_physics)

if __name__=='__main__':unittest.main()
