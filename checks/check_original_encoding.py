"""Two-event symbolic regression under the same updated physical parameters."""
from pathlib import Path
import sys,runpy
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R))
from context import *
from original_cpc import OriginalCPC
c=dict(configuration(),preview_gates=2,cells_per_leg=3)
before=runpy.run_path(str(R/'checks/original_cpc_before_preview.py'))['OriginalCPC'](c,course())
after=OriginalCPC(c,course())
for key in ('lbg','ubg','lbx','ubx'):np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
rng=np.random.default_rng(20261102);errors=[]
for _ in range(3):
    x=rng.normal(0,.1,before.solver.size1_in('x0'));x[-1]=4.
    p=rng.normal(0,.1,before.solver.size1_in('p'));p[-1]=1e-5
    for name in ('nlp_f','nlp_g'):
        a=np.array(before.solver.get_function(name)(x,p));b=np.array(after.solver.get_function(name)(x,p));errors.append(float(abs(a-b).max()))
np.testing.assert_allclose(errors,0.,atol=1e-10)
save_json(R/'checks/original_encoding_regression_20261102.json',dict(passed=True,parameterization='Two-gate original CPC',physical_parameters='Both expressions use updated drag and vehicle parameters',maximum_expression_difference=max(errors),samples=3,scope='Objective, constraints and bounds regression; no original-CPC full-flight feasibility claim.'))
print('Original two-event objective, constraints and bounds preserved:',max(errors))
