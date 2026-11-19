"""Eliminate the scaled shooting chain by block forward substitution.

The initial row and each next-state derivative are diagonal. The remaining
state dependence is the immediately preceding shooting node. No optimization
constraints are dropped; callers retain the complete linear residual check.
"""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np


def condense(A, dynamic, nx, upper, block_size=13):
    matrix=A[dynamic,:nx].tocoo()
    rows,cols=matrix.row,matrix.col
    block_rows,block_cols=rows//block_size,cols//block_size
    diagonal=rows==cols
    predecessor=block_rows==block_cols+1
    if len(dynamic)!=nx or nx%block_size or np.any(~(diagonal|predecessor)):
        raise ValueError('Dynamics matrix is not a block bidiagonal shooting chain')
    count=nx//block_size
    diag=np.zeros(nx)
    np.add.at(diag,rows[diagonal],matrix.data[diagonal])
    if np.any(np.abs(diag)<1e-14):
        raise ValueError('Singular shooting diagonal')
    blocks=np.zeros((count,block_size,block_size))
    np.add.at(blocks,(block_rows[predecessor],rows[predecessor]%block_size,
                     cols[predecessor]%block_size),matrix.data[predecessor])
    controls=A.shape[1]-nx
    rhs=np.column_stack((-A[dynamic,nx:].toarray(),upper[dynamic]))
    solution=rhs.reshape(count,block_size,controls+1).copy()
    diag=diag.reshape(count,block_size)
    solution[0]/=diag[0,:,None]
    for k in range(1,count):
        solution[k]=(solution[k]-blocks[k]@solution[k-1])/diag[k,:,None]
    solution=solution.reshape(nx,controls+1)
    return np.vstack((solution[:,:controls],np.eye(controls))),np.r_[solution[:,-1],np.zeros(controls)]
