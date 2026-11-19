"""Compare course progress in metres between ordered gate crossings in recorded races."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
from pathlib import Path
import numpy as np


def progress(folder):
    audit=json.loads((folder/'flight/spatial_audit.json').read_text())
    cross=np.array(audit['ordered_crossing_times_s'])
    if cross.shape!=(2,14):
        return None
    with np.load(folder/'flight/execution.npz') as f:
        t,x=f['time'],f['state']
    speed=np.linalg.norm(x[:,:,3:6],axis=2)
    length=np.vstack((np.zeros(2),np.cumsum(.5*(speed[1:]+speed[:-1])*np.diff(t)[:,None],axis=0)))
    stations=np.array([np.interp(np.r_[t[0],cross[i]],t,length[:,i]) for i in (0,1)])
    mean_station=stations.mean(0)
    along=np.column_stack([np.interp(length[:,i],stations[i],mean_station) for i in (0,1)])
    moving=(t<=cross[:,-1].min())&(speed.min(1)>2.)
    gap=abs(along[moving,0]-along[moving,1])
    return dict(n=int(moving.sum()),median_m=float(np.median(gap)),p90_m=float(np.quantile(gap,.9)),
                within_half_m=float(np.mean(gap<=.5)),within_one_m=float(np.mean(gap<=1.)))


if __name__=='__main__':
    import sys
    for name in sys.argv[1:]:
        print(name,progress(Path('reports/spatial_potential')/name))
