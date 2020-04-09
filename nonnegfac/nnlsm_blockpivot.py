# Nonnegativity Constrained Least Squares with Multiple Righthand Sides
#      using Block Principal Pivoting method
#
# This software solves the following problem: given A and B, find X such that
#              minimize || AX-B ||_F^2 where X>=0 elementwise.
#
# Reference:
#      Jingu Kim and Haesun Park, Toward Faster Nonnegative Matrix Factorization: A New Algorithm and Comparisons,
#      In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM'08), 353-362, 2008
#
# Written by Jingu Kim (jingu@cc.gatech.edu)
# Copyright 2008-2009 by Jingu Kim and Haesun Park,
#                        School of Computational Science and Engineering,
#                        Georgia Institute of Technology
#
# Check updated code at http://www.cc.gatech.edu/~jingu
# Please send bug reports, comments, or questions to Jingu Kim.
# This code comes with no guarantee or warranty of any kind. Note that this algorithm assumes that the
#      input matrix A has full column rank.
#
# Last modified Feb-20-2009
#
# <Inputs>
#        A : input matrix (m x n) (by default), or A'*A (n x n) if isInputProd==1
#        B : input matrix (m x k) (by default), or A'*B (n x k) if isInputProd==1
#        isInputProd : (optional, default:0) if turned on, use (A'*A,A'*B) as input instead of (A,B)
#        init : (optional) initial value for X
# <Outputs>
#        X : the solution (n x k)
#        Y : A'*A*X - A'*B where X is the solution (n x k)
#        iter : number of iterations
#        success : 1 for success, 0 for failure.
#                  Failure could only happen on a numericall very ill-conditioned problem.

import numpy as np

def nnlsm_blockpivot(A, B, isInputProd = None, init = None):
    if isInputProd is None and init is None:
        isInputProd = 0
    if isInputProd:
        AtA, AtB = A, B
    else:
        AtA, AtB = np.transpose(A) @ A, np.transpose(A) @ B

    n, k = AtB.shape
    MAX_ITER = n * 5
    # set initial feasible solution
    X = np.zeros(n, k)
    if isInputProd is not None and init is None:
        Y = - AtB
        PassiveSet = np.zeros(n, k)
        iter = 0
    else:
        PassiveSet = init > 0
        X, iter = solveNormalEqComb(AtA, AtB, PassiveSet)
        Y = AtA @ X - AtB
    # parameters
    pbar = 3
    P = np.zeros(1, k)
    P[:] = pbar
    Ninf = np.zeros(1, k)
    Ninf[:] = n + 1
    iter = 0

    NonOptSet = (Y < 0) & (not PassiveSet)
    InfeaSet = (X < 0) & PassiveSet
    NotGood = sum(NonOptSet) + sum(InfeaSet)
    NotOptCols = NotGood > 0

    bigIter = 0
    success = 1
    while np.count_nonzero(NotOptCols) > 0
        bigIter = bigIter+1
        if (MAX_ITER > 0) and (bigIter > MAX_ITER): # set max_iter for ill-conditioned (numerically unstable) case
            success = 0
            break

        Cols1 = NotOptCols & (NotGood < Ninf)
        Cols2 = NotOptCols & (NotGood >= Ninf) & (P >= 1)
        Cols3Ix = np.nonzero(NotOptCols & not Cols1 & not Cols2)
        if np.count_nonzero(Cols1) > 0:
            P[Cols1] = pbar
            Ninf[Cols1] = NotGood[Cols1]
            PassiveSet[NonOptSet & np.repmat(Cols1,n,1)] = True
            PassiveSet[InfeaSet & np.repmat(Cols1,n,1)] = False
        if np.count_nonzero(Cols2) > 0:
            P[Cols2] = P[Cols2]-1
            PassiveSet[NonOptSet & np.repmat(Cols2,n,1)] = True
            PassiveSet[InfeaSet & np.repmat(Cols2,n,1)] = False
        if np.count_nonzero(NotOptCols & not Cols1 & not Cols2) > 0:
            for i in range(length(Cols3Ix)):
                Ix = Cols3Ix[i]
                toChange = max(find( NonOptSet[:, Ix] | InfeaSet[:,Ix] ))
                if PassiveSet[toChange,Ix]:
                    PassiveSet[toChange,Ix] = False
                else:
                    PassiveSet[toChange,Ix] = True
        NotOptMask = np.repmat(NotOptCols,n,1)
        X[:,NotOptCols], subiter = solveNormalEqComb(AtA, AtB[:,NotOptCols], PassiveSet[:,NotOptCols])
        iter = iter + subiter
        X[abs(X) < 1e-12] = 0            # for numerical stability
        Y[:,NotOptCols] = AtA @ X[:,NotOptCols] - AtB[:,NotOptCols]
        Y[abs(Y) < 1e-12] = 0            # for numerical stability

        # check optimality
        NonOptSet = NotOptMask & (Y < 0) & not PassiveSet
        InfeaSet = NotOptMask & (X < 0) & PassiveSet
        NotGood = sum(NonOptSet) + sum(InfeaSet)
        NotOptCols = NotGood > 0
    return X,Y,iter,success