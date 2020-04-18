import time
from math import sqrt

import numpy as np
import scipy
from numpy.linalg import norm


def solveNormalEqComb(AtA, AtB, PassSet=None):
    # Solve normal equations using combinatorial grouping.
    # Although this function was originally adopted from the code of
    # "M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450",
    # important modifications were made to fix bugs.
    #
    # Modified by Jingu Kim (jingu@cc.gatech.edu)
    #             School of Computational Science and Engineering,
    #             Georgia Institute of Technology
    #
    # Last updated Aug-12-2009

    iter = 0
    if PassSet is None or np.count_nonzero(PassSet) == 0 or np.count_nonzero(PassSet == 0) == 0:
        Z = np.linalg.solve(AtA, AtB)
        iter = iter + 1
    else:
        Z = np.zeros(AtB.shape)
        if len(PassSet.shape) == 1:
            n = PassSet.shape
            k1 = 1
        else:
            n, k1 = PassSet.shape

        ## Fixed on Aug-12-2009
        if k1 == 1:
            PassSet = PassSet.reshape(-1)
            Z[PassSet] = np.linalg.solve(AtA[np.ix_(PassSet, PassSet)], AtB[PassSet])
        else:
            ## Fixed on Aug-12-2009
            # The following bug was identified by investigating a bug report by Hanseung Lee.
            sortedPassSet = sorted(enumerate(PassSet.T.tolist()), key=lambda x: x[1])
            sortIx = [x[0] for x in sortedPassSet]
            sortedPassSet = [x[1] for x in sortedPassSet]
            breakIx = [0, *np.where(np.diff(sortedPassSet, axis=0).T.any(axis=0))[0], k1]

            for k in range(len(breakIx) - 1):
                cols = sortIx[(breakIx[k] + 1): breakIx[k + 1]]
                vars = PassSet[:, sortIx[breakIx[k] + 1]]
                Z[np.ix_(vars, cols)] = np.linalg.solve(AtA[np.ix_(vars, vars)], AtB[np.ix_(vars, cols)])
                iter += 1
    return Z, iter


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

def nnlsm_activeset(A, B, overwrite=None, isInputProd=None, init=None):
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
        NotOptSet = np.any(X < 0, axis=0)
    else:
        if init is None:
            X = np.zeros(n, k)
            PassSet = np.zeros(n, k)
        else:
            X = init
            PassSet = X > 0
        NotOptSet = np.ones(1, k)
        iter = 0

    Y = np.zeros(n, k)
    Y[:, ~NotOptSet] = AtA @ X[:, ~NotOptSet] - AtB[:, ~NotOptSet]
    NotOptCols = np.nonzero(NotOptSet)

    bigIter, success = 0, 1
    while len(NotOptCols) > 0:
        bigIter += 1
        if (MAX_ITER > 0) and (bigIter > MAX_ITER):  # set max_iter for ill-conditioned (numerically unstable) case
            success = 0
            break

        # find unconstrained LS solution for the passive set
        Z, subiter = solveNormalEqComb(AtA, AtB[:, NotOptCols], PassSet[:, NotOptCols])
        iter += subiter
        InfeaSubSet = Z < 0
        InfeaSubCols = np.where(InfeaSubSet.any(axis=0))[0]
        FeaSubCols = np.where((~InfeaSubSet).all(axis=0))[0]

        if len(InfeaSubCols) > 0:  # for infeasible cols
            ZInfea = Z[:, InfeaSubCols]
            InfeaCols = NotOptCols[InfeaSubCols]
            Alpha = np.zeros(n, len(InfeaSubCols))
            Alpha[:, :] = np.inf
            i, j = np.nonzero(InfeaSubSet[:, InfeaSubCols])
            InfeaSubIx = np.ravel_multi_index((i, j), Alpha.shape, order='F')
            if len(InfeaCols) == 1:
                InfeaIx = np.ravel_multi_index((i, InfeaCols @ np.ones(len(j), 1)), [n, k], order='F')
            else:
                InfeaIx = np.ravel_multi_index((i, InfeaCols(j).T), [n, k], order='F')
            Alpha[InfeaSubIx] = X[InfeaIx] / (X[InfeaIx] - ZInfea[InfeaSubIx])

            minIx = np.argmin(Alpha)
            minVal = Alpha[minIx]
            Alpha[:, :] = np.matlib.repmat(minVal, n, 1)
            X[:, InfeaCols] = X[:, InfeaCols] + Alpha * (ZInfea - X[:, InfeaCols])
            IxToActive = np.ravel_multi_index((minIx, InfeaCols), [n, k], order="F")
            X[IxToActive] = 0
            PassSet[IxToActive] = False

        if len(FeaSubCols) > 0:  # for feasible cols
            FeaCols = NotOptCols[FeaSubCols]
            X[:, FeaCols] = Z[:, FeaSubCols]
            Y[:, FeaCols] = AtA @ X[:, FeaCols] - AtB[:, FeaCols]
            # Y( abs(Y)<1e-12 ) = 0               # One can uncomment this line for numerical stability.

            NotOptSubSet = (Y[:, FeaCols] < 0) & ~PassSet[:, FeaCols]
            NewOptCols = FeaCols[np.all(~NotOptSubSet, axis=0)]
            UpdateNotOptCols = FeaCols[np.any(NotOptSubSet, axis=0)]
            if len(UpdateNotOptCols) > 0:
                temp = Y[:, UpdateNotOptCols] * ~PassSet[:, UpdateNotOptCols]
                minIx = np.argmin(temp)
                minVal = temp[minIx]
                PassSet[np.ravel_multi_index((minIx, UpdateNotOptCols), [n, k], order="F")] = True
            NotOptSet[NewOptCols] = False
            NotOptCols = np.nonzero(NotOptSet)
    return X, Y, iter, success


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

def nnlsm_blockpivot(A, B, isInputProd, init):
    if isInputProd:
        AtA, AtB = A, B
    else:
        AtA, AtB = A.T @ A, A.T @ B
    n, k = AtB.shape
    MAX_ITER = n * 5
    # set initial feasible solution
    X = np.zeros((n, k))
    PassiveSet = init > 0
    X, iter = solveNormalEqComb(AtA, AtB, PassiveSet)
    Y = AtA @ X - AtB

    # parameters
    pbar = 3
    P = np.zeros((1, k))
    P[:] = pbar
    Ninf = np.zeros((1, k))
    Ninf[:] = n + 1
    iter = 0

    PassiveSet = PassiveSet.reshape(len(Y), -1)
    NonOptSet = (Y < 0) & ~PassiveSet
    InfeaSet = (X < 0) & PassiveSet
    NotGood = (np.sum(NonOptSet, axis=0) + np.sum(InfeaSet, axis=0)).reshape(1, -1)
    NotOptCols = (NotGood > 0).reshape(-1)

    bigIter = 0
    success = 1
    while np.count_nonzero(NotOptCols) > 0:
        bigIter = bigIter + 1
        if (MAX_ITER > 0) and (bigIter > MAX_ITER):  # set max_iter for ill-conditioned (numerically unstable) case
            success = 0
            break

        Cols1 = np.squeeze(NotOptCols & (NotGood < Ninf))
        Cols2 = NotOptCols & (NotGood >= Ninf) & (P >= 1)
        Cols3Ix = np.nonzero(NotOptCols & ~Cols1 & ~Cols2)[1]
        if np.count_nonzero(Cols1) > 0:
            P[0, Cols1] = pbar
            Ninf[0, Cols1] = NotGood[0, Cols1]
            PassiveSet[NonOptSet & np.matlib.repmat(Cols1, n, 1)] = True
            PassiveSet[InfeaSet & np.matlib.repmat(Cols1, n, 1)] = False
        if np.count_nonzero(Cols2) > 0:
            P[Cols2] = P[Cols2] - 1
            PassiveSet[NonOptSet & np.matlib.repmat(Cols2, n, 1)] = True
            PassiveSet[InfeaSet & np.matlib.repmat(Cols2, n, 1)] = False
        if np.count_nonzero(NotOptCols & ~Cols1 & ~Cols2) > 0:
            for i in range(len(Cols3Ix)):
                Ix = Cols3Ix[i]
                toChange = np.amax(np.nonzero(NonOptSet[:, Ix] | InfeaSet[:, Ix]))
                if PassiveSet[toChange, Ix]:
                    PassiveSet[toChange, Ix] = False
                else:
                    PassiveSet[toChange, Ix] = True
        NotOptMask = np.matlib.repmat(NotOptCols, n, 1)
        NotOptCols = NotOptCols.reshape(-1)
        X[:, NotOptCols], subiter = solveNormalEqComb(AtA, AtB[:, NotOptCols], PassiveSet[:, NotOptCols])
        iter = iter + subiter
        X[abs(X) < 1e-12] = 0  # for numerical stability
        Y[:, NotOptCols] = AtA @ X[:, NotOptCols] - AtB[:, NotOptCols]
        Y[abs(Y) < 1e-12] = 0  # for numerical stability

        # check optimality
        NonOptSet = NotOptMask & (Y < 0) & ~PassiveSet
        InfeaSet = NotOptMask & (X < 0) & PassiveSet
        NotGood = (np.sum(NonOptSet, axis=0) + np.sum(InfeaSet, axis=0)).reshape(1, -1)
        NotOptCols = NotGood > 0
    return X, Y, iter, success


# Nonnegative Matrix Factorization by Alternating Nonnegativity Constrained Least Squares
#      using Block Principal Pivoting/Active Set method
#
# This software solves one the following problems: given A and k, find W and H such that
#     (1) minimize 1/2 * || A-WH ||_F^2
#     (2) minimize 1/2 * ( || A-WH ||_F^2 + alpha * || W ||_F^2 + beta * || H ||_F^2 )
#     (3) minimize 1/2 * ( || A-WH ||_F^2 + alpha * || W ||_F^2 + beta * (sum_(i=1)^n || H(:,i) ||_1^2 ) )
#     where W>=0 and H>=0 elementwise.
#
# Reference:
#  [1] For using this software, please cite:
#          Jingu Kim and Haesun Park, Toward Faster Nonnegative Matrix Factorization: A New Algorithm and Comparisons,
#                 In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM'08), 353-362, 2008
#  [2] If you use 'nnls_solver'='as' (see below), please cite:
#          Hyunsoo Kim and Haesun Park, Nonnegative Matrix Factorization Based on Alternating Nonnegativity Constrained Least Squares and Active Set Method,
#                 SIAM Journal on Matrix Analysis and Applications, 2008, 30, 713-730
#
# Written by Jingu Kim (jingu@cc.gatech.edu)
# Copyright 2008-2009 by Jingu Kim and Haesun Park,
#                        School of Computational Science and Engineering,
#                        Georgia Institute of Technology
#
# Check updated code at http://www.cc.gatech.edu/~jingu
# Please send bug reports, comments, or questions to Jingu Kim.
# This code comes with no guarantee or warranty of any kind.
#
# Last modified Feb-20-2010
#
# <Inputs>
#        A : Input data matrix (m x n)
#        k : Target low-rank
#
#        (Below are optional arguments: can be set by providing name-value pairs)
#        TYPE : 'plain' to use formulation (1)
#               'regularized' to use formulation (2)
#               'sparse' to use formulation (3)
#               Default is 'regularized', which is recommended for quick application testing unless 'sparse' or 'plain' is explicitly needed.
#               If sparsity is needed for 'W' factor, then apply this function for the transpose of 'A' with formulation (3).
#                      Then, exchange 'W' and 'H' and obtain the transpose of them.
#               Imposing sparsity for both factors is not recommended and thus not included in this software.
#        NNLS_SOLVER : 'bp' to use the algorithm in [1]
#                      'as' to use the algorithm in [2]
#                      Default is 'bp', which is in general faster.
#        ALPHA : Parameter alpha in the formulation (2) or (3).
#                Default is the average of all elements in A. No good justfication for this default value, and you might want to try other values.
#        BETA : Parameter beta in the formulation (2) or (3).
#               Default is the average of all elements in A. No good justfication for this default value, and you might want to try other values.
#        MAX_ITER : Maximum number of iterations. Default is 100.
#        MIN_ITER : Minimum number of iterations. Default is 20.
#        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
#        W_INIT : (m x k) initial value for W.
#        H_INIT : (k x n) initial value for H.
#        TOL : Stopping tolerance. Default is 1e-3. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
#        VERBOSE : 0 (default) - No debugging information is collected.
#                  1 (debugging purpose) - History of computation is returned by 'HIS' variable.
#                  2 (debugging purpose) - History of computation is additionally printed on screen.
# <Outputs>
#        W : Obtained basis matrix (m x k)
#        H : Obtained coefficients matrix (k x n)
#        iter : Number of iterations
#        HIS : (debugging purpose) History of computation
# <Usage Examples>
#        nmf(A,10)
#        nmf(A,20,'verbose',2)
#        nmf(A,30,'verbose',2,'nnls_solver','as')
#        nmf(A,5,'verbose',2,'type','sparse')
#        nmf(A,60,'verbose',1,'type','plain','w_init',rand(m,k))
#        nmf(A,70,'verbose',2,'type','sparse','nnls_solver','bp','alpha',1.1,'beta',1.3)

class Par(object):
    """docstring for par"""

    def __init__(self, m, n, type_, nnls_solver, alpha, beta, max_iter, min_iter, max_time, tol, verbose):
        super(Par, self).__init__()
        self.m = m
        self.n = n
        self.type = type_
        self.nnls_solver = nnls_solver
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_time = max_time
        self.tol = tol
        self.verbose = verbose


class Ver(object):
    """docstring for Ver"""

    def __init__(self, initGrNormW, initGrNormH, initNorm, SC1, SC2, SC3, W_density, H_density):
        super(Ver, self).__init__()
        self.initGrNormW = initGrNormW
        self.initGrNormH = initGrNormH
        self.initNorm = initNorm
        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.W_density = W_density
        self.H_density = H_density


class Final(object):
    def __init__(self, par, iter, tTotal, tStart, A, W, H, m, n, k):
        super(Final, self).__init__()
        self.iterations = iter
        if par.verbose:
            self.elapsed_total = tTotal
        else:
            self.elapsed_total = time.time() - tStart
        self.relative_error = np.linalg.norm(A - W @ H, 'fro') / np.linalg.norm(A, 'fro')
        self.W_density = np.count_nonzero(W > 0) / (m * k)
        self.H_density = np.count_nonzero(H > 0) / (n * k)


def nmf(A, k, varargin):
    m, n = A.shape
    ST_RULE = 1

    # Default configuration
    par = Par(m, n, 'regularized', 'bp', 0, 0, 100, 20, 1e6, 1e-3, 0)

    W = np.random.rand(m, k)
    H = np.random.rand(k, n)

    # Read optional parameters
    if len(varargin) % 2 == 1:
        exit('Optional parameters should always go by pairs')
    else:
        for i in range(0, len(varargin) - 1, 2):
            ups = varargin[i].upper()
            if ups == 'TYPE':
                par.type = varargin[i + 1]
            elif ups == 'NNLS_SOLVER':
                par.nnls_solver = varargin[i + 1]
            elif ups == 'ALPHA':
                argAlpha = varargin[i + 1]
                par.alpha = argAlpha
            elif ups == 'BETA':
                argBeta = varargin[i + 1]
                par.beta = argBeta
            elif ups == 'MAX_ITER':
                par.max_iter = varargin[i + 1]
            elif ups == 'MIN_ITER':
                par.min_iter = varargin[i + 1]
            elif ups == 'MAX_TIME':
                par.max_time = varargin[i + 1]
            elif ups == 'W_INIT':
                W = varargin[i + 1]
            elif ups == 'H_INIT':
                H = varargin[i + 1]
            elif ups == 'TOL':
                par.tol = varargin[i + 1]
            elif ups == 'VERBOSE':
                par.verbose = varargin[i + 1]
            else:
                exit(['Unrecognized option: ', varargin[i]])

    # for regularized/sparse case
    if par.type == 'regularized':
        if 'argAlpha' not in locals():
            par.alpha = np.mean(A[:])
        if 'argBeta' not in locals():
            par.beta = np.mean(A[:])
        salphaI = sqrt(par.alpha) * np.eye(k)
        sbetaI = sqrt(par.beta) * np.eye(k)
        zerokn = np.zeros(k, n)
        zerokm = np.zeros(k, m)
    elif par.type == 'sparse':
        if 'argAlpha' not in locals():
            par.alpha = np.mean(A[:])
        if 'argBeta' not in locals():
            par.beta = np.mean(A[:])
        salphaI = sqrt(par.alpha) * np.eye(k)
        sbetaE = sqrt(par.beta) * np.ones(1, k)
        betaI = par.beta * np.ones(k, k)
        zero1n = np.zeros(1, n)
        zerokm = np.zeros(k, m)
    elif par.type != 'plain':
        exit(['Unrecognized type: use ''plain'', ''regularized'', or ''sparse''.'])

    if par.nnls_solver != 'bp' and par.nnls_solver != 'as':
        exit(['Unrecognized nnls_solver: use ''bp'' or ''as''.'])

    print(par)

    HIS = 0
    if par.verbose:  # collect information for analysis/debugging
        gradW, gradH = getGradient(A, W, H, par.type, par.alpha, par.beta)
        initGrNormW = np.linarg.norm(gradW, 'fro')
        initGrNormH = np.linarg.norm(gradH, 'fro')
        initNorm = np.linarg.norm(A, 'fro')
        numSC = 3
        initSCs = np.zeros(numSC, 1)
        for j in range(numSC):
            initSCs[j] = getInitCriterion(j, A, W, H, par.type, par.alpha, par.beta, gradW, gradH)
        # ---(1)------(2)--------(3)--------(4)--------(5)---------(6)----------(7)------(8)-----(9)-------(10)--------------(11)-------
        # iter # | elapsed | totalTime | subIterW | subIterH | rel. obj.(#) | NM_GRAD | GRAD | DELTA | W density (#) | H density (#)
        # ------------------------------------------------------------------------------------------------------------------------------
        HIS = np.zeros(1, 11)
        HIS[1, 0:5] = 0
        ver = Ver(initGrNormW, initGrNormH, initNorm, initSCs[1], initSCs[2], initSCs[3],
                  len(np.nonzero(W > 0)) / (m * k), len(np.nonzero(H > 0)) / (n * k))
        HIS[1, 6] = ver.initNorm
        HIS[1, 7] = ver.SC1
        HIS[1, 8] = ver.SC2
        HIS[1, 9] = ver.SC3
        HIS[1, 10] = ver.W_density
        HIS[1, 11] = ver.H_density
        if par.verbose == 2:
            print(ver)
        tPrev = time.time()

    tStart = time.time()
    tTotal = 0
    initSC = getInitCriterion(ST_RULE, A, W, H, par.type, par.alpha, par.beta)
    SCconv = 0
    SC_COUNT = 3

    for iter in range(par.max_iter):
        if par.type == "plain":
            H, gradHX, subIterH = nnlsm(W, A, H, par.nnls_solver)
            W, gradW, subIterW = nnlsm(H.T, A.T, W.T, par.nnls_solver)
            W = W.T
            gradW = gradW.T
            gradH = (W.T @ W) @ H - W.T @ A
        elif par.type == 'regularized':
            H, gradHX, subIterH = nnlsm([W, sbetaI], [A, zerokn], H, par.nnls_solver)
            W, gradW, subIterW = nnlsm([H.T, salphaI], [A.T, zerokm], W.T, par.nnls_solver)
            W, gradW = W.T, gradW.T
            gradH = (W.T @ W) @ H - W.T @ A + par.beta * H
        elif par.type == 'sparse':
            H, gradHX, subIterH = nnlsm([W, sbetaE], [A, zero1n], H, par.nnls_solver)
            W, gradW, subIterW = nnlsm([H.T, salphaI], [A.T, zerokm], W.T, par.nnls_solver)
            W, gradW = W.T, gradW.T
            gradH = (W.T @ W) @ H - W.T @ A + betaI @ H

        if par.verbose:  # collect information for analysis/debugging
            elapsed = time.time() - tPrev
            tTotal = tTotal + elapsed
            ver = 0
            idx = iter + 1
            # ---(1)------(2)--------(3)--------(4)--------(5)---------(6)----------(7)------(8)-----(9)-------(10)--------------(11)-------
            # iter # | elapsed | totalTime | subIterW | subIterH | rel. obj.(#) | NM_GRAD | GRAD | DELTA | W density (#) | H density (#)
            # ------------------------------------------------------------------------------------------------------------------------------
            ver.iter = iter
            ver.elapsed = elapsed
            ver.tTotal = tTotal
            ver.subIterW = subIterW
            ver.subIterH = subIterH
            ver.relError = norm(A - W * H, 'fro') / initNorm
            ver.SC1 = getStopCriterion(1, A, W, H, par.type, par.alpha, par.beta, gradW, gradH) / initSCs(1)
            ver.SC2 = getStopCriterion(2, A, W, H, par.type, par.alpha, par.beta, gradW, gradH) / initSCs(2)
            ver.SC3 = getStopCriterion(3, A, W, H, par.type, par.alpha, par.beta, gradW, gradH) / initSCs(3)
            ver.W_density = np.count_nonzero(W > 0) / (m * k)
            ver.H_density = np.count_nonzero(H > 0) / (n * k)
            HIS[idx, 1] = iter
            HIS[idx, 2] = elapsed
            HIS[idx, 3] = tTotal
            HIS[idx, 4] = subIterW
            HIS[idx, 5] = subIterH
            HIS[idx, 6] = ver.relError
            HIS[idx, 7] = ver.SC1
            HIS[idx, 8] = ver.SC2
            HIS[idx, 9] = ver.SC3
            HIS[idx, 10] = ver.W_density
            HIS[idx, 11] = ver.H_density
            if par.verbose == 2:
                print(ver)
            tPrev = time.time()

        if (iter > par.min_iter):
            SC = getStopCriterion(ST_RULE, A, W, H, par.type, par.alpha, par.beta, gradW, gradH)
            if (par.verbose and (tTotal > par.max_time)) or (~par.verbose and ((time.time() - tStart) > par.max_time)):
                break
            elif (SC / initSC <= par.tol):
                SCconv = SCconv + 1
                if (SCconv >= SC_COUNT):
                    break
            else:
                SCconv = 0
    m, n = A.shape
    norm2 = sqrt(np.sum(np.square(W), axis=0))
    toNormalize = norm2 > 0
    W[:, toNormalize] = W[:, toNormalize] / np.matlib.repmat(norm2[toNormalize], m, 1)
    H[toNormalize, :] = H[toNormalize, :] * np.matlib.repmat(norm2[toNormalize].T, 1, n)

    final = Final(par, iter, tTotal, tStart, A, W, H, m, n, k)
    print(final)
    return W, H, iter, HIS


# ------------------------------------------------------------------------------------------------------------------------
#                                    Utility Functions
# ------------------------------------------------------------------------------------------------------------------------
def nnlsm(A, B, init, solver):
    '''solver == "bp" or "as"'''
    return nnlsm_blockpivot(A, B, 0, init) if solver == "bp" else nnlsm_activeset(A, B, 1, 0, init)


# -------------------------------------------------------------------------------
def getInitCriterion(stopRule, A, W, H, type, alpha, beta, gradW=None, gradH=None):
    # STOPPING_RULE : 1 - Normalized proj. gradient
    #                 2 - Proj. gradient
    #                 3 - Delta by H. Kim
    #                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if gradW is None or gradH is None:
        gradW, gradH = getGradient(A, W, H, type, alpha, beta)
    m, k = W.shape
    k, n = H.shape
    numAll = m * k + k * n
    if stopRule == 1:
        retVal = np.linalg.norm([gradW, gradH.T], 'fro') / numAll
    elif stopRule == 2:
        retVal = np.linalg.norm([gradW, gradH.T], 'fro')
    elif stopRule == 3:
        retVal = getStopCriterion(3, A, W, H, type, alpha, beta, gradW, gradH)
    elif stopRule == 0:
        retVal = 1
    return retVal


# -------------------------------------------------------------------------------
def getStopCriterion(stopRule, A, W, H, type, alpha, beta, gradW=None, gradH=None):
    # STOPPING_RULE : 1 - Normalized proj. gradient
    #                 2 - Proj. gradient
    #                 3 - Delta by H. Kim
    #                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if gradW is None or gradH is None:
        gradW, gradH = getGradient(A, W, H, type, alpha, beta)

    if stopRule == 1:
        pGradW = gradW[gradW < 0 | W > 0]
        pGradH = gradH[gradH < 0 | H > 0]
        pGrad = [gradW[gradW < 0 | W > 0], gradH[gradH < 0 | H > 0]]
        pGradNorm = np.linalg.norm(pGrad)
        retVal = pGradNorm / len(pGrad)
    elif stopRule == 2:
        pGradW = gradW[gradW < 0 | W > 0]
        pGradH = gradH[gradH < 0 | H > 0]
        pGrad = [gradW[gradW < 0 | W > 0], gradH[gradH < 0 | H > 0]]
        retVal = np.linalg.norm(pGrad)
    elif stopRule == 3:
        resmat = min(H, gradH)
        resvec = resmat[:]
        resmat = min(W, gradW)
        resvec = [resvec, resmat[:]]
        deltao = np.linalg.norm(resvec, 1)  # L1-norm
        num_notconv = np.count_nonzero(abs(resvec) > 0)
        retVal = deltao / num_notconv
    elif stopRule == 0:
        retVal = 1e100
    return retVal


# -------------------------------------------------------------------------------
def getGradient(A, W, H, type, alpha, beta):
    if type == "plain":
        gradW = W @ (H @ H.T) - A @ H.T
        gradH = (W.T @ W) @ H - W.T @ A
    elif type == 'regularized':
        gradW = W @ (H @ H.T) - A @ H.T + alpha * W
        gradH = (W.T @ W) @ H - W.T @ A + beta * H
    elif type == 'sparse':
        k = W.shape[1]
        betaI = beta * np.ones(k, k)
        gradW = W @ (H @ H.T) - A @ H.T + alpha * W
        gradH = (W.T @ W) @ H - W.T @ A + betaI * H
    return gradW, gradH
