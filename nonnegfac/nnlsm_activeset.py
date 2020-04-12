# Nonnegativity Constrained Least Squares with Multiple Righthand Sides
#      using Active Set method
#
# This software solves the following problem: given A and B, find X such that
#            minimize || AX-B ||_F^2 where X>=0 elementwise.
#
# Reference:
#      Charles L. Lawson and Richard J. Hanson, Solving Least Squares Problems,
#            Society for Industrial and Applied Mathematics, 1995
#      M. H. Van Benthem and M. R. Keenan,
#            Fast Algorithm for the Solution of Large-scale Non-negativity-constrained Least Squares Problems,
#            J. Chemometrics 2004; 18: 441-450
#
# Written by Jingu Kim (jingu@cc.gatech.edu)
#               School of Computational Science and Engineering,
#               Georgia Institute of Technology
#
# Last updated Apr-02-2012
#
# <Inputs>
#        A : input matrix (m x n) (by default), or A'*A (n x n) if isInputProd==1
#        B : input matrix (m x k) (by default), or A'*B (n x k) if isInputProd==1
#        overwrite : (optional, default:0) if turned on, unconstrained least squares solution is computed in the beginning
#        isInputProd : (optional, default:0) if turned on, use (A'*A,A'*B) as input instead of (A,B)
#        init : (optional) initial value for X
# <Outputs>
#        X : the solution (n x k)
#        Y : A'*A*X - A'*B where X is the solution (n x k)
#        iter : number of iterations
#        success : 1 for success, 0 for failure.
#                  Failure could only happen on a numericall very ill-conditioned problem.

import numpy as np
def nnlsm_activeset( A, B, overwrite = 0, isInputProd = 0, init):
    if isInputProd:
        AtA, AtB = A, B
    else:
        AtA, AtB = A.T @ A, A.T @ B

    n, k = AtB.shape
    MAX_ITER = n * 5
    # set initial feasible solution
    if overwrite:
        X, iter = solveNormalEqComb(AtA, AtB)
        PassSet = X > 0
        NotOptSet = np.any(X < 0, axis = 0)
    else:
        if nargin < 5:
            X = np.zeros(n,k)
            PassSet = np.zeros(n,k)
        else:
            X = init
            PassSet = X > 0
        NotOptSet = np.ones(1, k)
        iter = 0

    Y = np.zeros(n, k)
    Y[:, ~NotOptSet] = AtA @ X[:,~NotOptSet] - AtB[:,~NotOptSet]
    NotOptCols = np.nonzero(NotOptSet)

    bigIter, success = 0, 1
    while len(NotOptCols) > 0:
        bigIter += 1
        if (MAX_ITER > 0) and (bigIter > MAX_ITER):   # set max_iter for ill-conditioned (numerically unstable) case
            success = 0
            break

        # find unconstrained LS solution for the passive set
        Z, subiter = solveNormalEqComb(AtA,AtB[:,NotOptCols],PassSet[:,NotOptCols])
        iter += subiter
        InfeaSubSet = Z < 0
        InfeaSubCols = np.where(InfeaSubSet.any(axis = 0))[0]
        FeaSubCols = np.where((~InfeaSubSet).all(axis = 0))[0]

        if len(InfeaSubCols) > 0:               # for infeasible cols
            ZInfea = Z[:,InfeaSubCols]
            InfeaCols = NotOptCols[InfeaSubCols]
            Alpha = np.zeros(n, len(InfeaSubCols))
            Alpha[:,:] = Inf
            i,j = np.nonzero(InfeaSubSet[:,InfeaSubCols])
            InfeaSubIx = np.ravel_multi_index((i,j), Alpha.shape, order = 'F')
            if len(InfeaCols) == 1:
                InfeaIx = np.ravel_multi_index((i, InfeaCols @ np.ones(length(j),1)), [n, k], order = 'F')
            else:
                InfeaIx = np.ravel_multi_index((i, InfeaCols(j).T), [n, k], order = 'F')
            Alpha[InfeaSubIx] = X[InfeaIx] / (X[InfeaIx]-ZInfea[InfeaSubIx])

            minIx = np.argmin(Alpha)
            minVal = Alpha[minIx]
            Alpha[:,:] = np.matlib.repmat(minVal,n,1)
            X[:,InfeaCols] = X[:,InfeaCols] + Alpha * (ZInfea - X[:,InfeaCols])
            IxToActive = np.ravel_multi_index((minIx,InfeaCols), [n, k], order = "F")
            X[IxToActive] = 0
            PassSet[IxToActive] = False

        if len(FeaSubCols)>0:                 # for feasible cols
            FeaCols = NotOptCols[FeaSubCols]
            X[:,FeaCols] = Z[:,FeaSubCols]
            Y[:,FeaCols] = AtA @ X[:,FeaCols] - AtB[:,FeaCols]
            #Y( abs(Y)<1e-12 ) = 0               # One can uncomment this line for numerical stability.

            NotOptSubSet = (Y[:,FeaCols] < 0) & ~PassSet[:,FeaCols]
            NewOptCols = FeaCols[np.all(~NotOptSubSet, axis = 0)]
            UpdateNotOptCols = FeaCols[np.any(NotOptSubSet, axis = 0)]
            if len(UpdateNotOptCols) > 0:
                temp = Y[:,UpdateNotOptCols] * ~PassSet[:,UpdateNotOptCols]
                minIx = np.argmin(temp)
                minVal = temp[minIx]
                PassSet[np.ravel_multi_index((minIx,UpdateNotOptCols), [n,k], order = "F")] = True
            NotOptSet[NewOptCols] = False
            NotOptCols = np.nonzero(NotOptSet)
    return X,Y,iter,success
