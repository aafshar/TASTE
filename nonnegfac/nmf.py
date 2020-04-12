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
import numpy as np

class Par(object):
    """docstring for par"""
    def __init__(self, m, n, type, nnls_solver, alpha, beta, max_iter, min_iter, max_time, tol, verbose):
        super(par, self).__init__()
        self.m = m
        self.n = n
        self.type = type
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

def nmf(A,k,varargin):
    m,n = A.shape
    ST_RULE = 1

    # Default configuration
    par = Par(m, n, 'regularized', 'bp', 0, 0, 100, 20, 1e6, 1e-3, 0)

    W = np.random.rand(m,k)
    H = np.random.rand(k,n)

    # Read optional parameters
    if len(varargin) % 2 == 1:
        error('Optional parameters should always go by pairs')
    else:
        for i in range(0, length(varargin)-1, 2):
            ups = varargin[i].upper()
            if ups == 'TYPE':
                par.type = varargin[i+1]
            elif ups == 'NNLS_SOLVER':
                par.nnls_solver = varargin[i+1]
            elif ups == 'ALPHA':
                argAlpha = varargin[i+1]
                par.alpha = argAlpha
            elif ups == 'BETA':
                argBeta = varargin[i+1]
                par.beta = argBeta
            elif ups == 'MAX_ITER':
                par.max_iter = varargin[i+1]
            elif ups == 'MIN_ITER':
                par.min_iter = varargin[i+1]
            elif ups == 'MAX_TIME':
                par.max_time = varargin[i+1]
            elif ups == 'W_INIT':
                W = varargin[i+1]
            elif ups == 'H_INIT':
                H = varargin[i+1]
            elif ups == 'TOL':
                par.tol = varargin[i+1]
            elif ups == 'VERBOSE':
                par.verbose = varargin[i+1]
            else:
                error(['Unrecognized option: ',varargin[i]])

    # for regularized/sparse case
    if par.type == 'regularized':
        if 'argAlpha' not in locals():
            par.alpha = mean(A[:])
        if 'argBeta' not in locals():
            par.beta = mean(A[:])
        salphaI = sqrt(par.alpha) * np.eye(k)
        sbetaI = sqrt(par.beta) * np.eye(k)
        zerokn = np.zeros(k,n)
        zerokm = np.zeros(k,m)
    elif par.type == 'sparse':
        if 'argAlpha' not in locals():
            par.alpha = mean(A[:])
        if 'argBeta' not in locals():
            par.beta = mean(A[:])
        salphaI = sqrt(par.alpha) * np.eye(k)
        sbetaE = sqrt(par.beta) * np.ones(1,k)
        betaI = par.beta * np.ones(k,k)
        zero1n = np.zeros(1,n)
        zerokm = np.zeros(k,m)
    elif par.type != 'plain':
        error(['Unrecognized type: use ''plain'', ''regularized'', or ''sparse''.'])

    if par.nnls_solver != 'bp' and par.nnls_solver != 'as'
        error(['Unrecognized nnls_solver: use ''bp'' or ''as''.'])

    print(par)

    HIS = 0
    if par.verbose:          # collect information for analysis/debugging
        gradW, gradH = getGradient(A,W,H,par.type,par.alpha,par.beta)
        initGrNormW = np.linarg.norm(gradW,'fro')
        initGrNormH = np.linarg.norm(gradH,'fro')
        initNorm = np.linarg.norm(A,'fro')
        numSC = 3
        initSCs = np.zeros(numSC,1)
        for j in range(numSC):
            initSCs[j] = getInitCriterion(j,A,W,H,par.type,par.alpha,par.beta,gradW,gradH)
#---(1)------(2)--------(3)--------(4)--------(5)---------(6)----------(7)------(8)-----(9)-------(10)--------------(11)-------
# iter # | elapsed | totalTime | subIterW | subIterH | rel. obj.(#) | NM_GRAD | GRAD | DELTA | W density (#) | H density (#)
#------------------------------------------------------------------------------------------------------------------------------
        HIS = np.zeros(1,11)
        HIS[1,0:5] = 0
        ver = Ver(initGrNormW, initGrNormH, initNorm, initSCs[1], initSCs[2], initSCs[3], length(np.nonzero(W>0)) / (m * k), length(np.nonzero(H>0)) / (n * k))
        HIS(1,6)=ver.initNorm
        HIS(1,7)=ver.SC1
        HIS(1,8)=ver.SC2
        HIS(1,9)=ver.SC3
        HIS(1,10)=ver.W_density
        HIS(1,11)=ver.H_density
        if par.verbose == 2:
            print(ver)
        tPrev = cputime

    tStart = cputime
    tTotal = 0
    initSC = getInitCriterion(ST_RULE,A,W,H,par.type,par.alpha,par.beta)
    SCconv = 0 SC_COUNT = 3

    for iter in range(par.max_iter):
        if par.type == "plain":
            H,gradHX,subIterH = nnlsm(W,A,H,par.nnls_solver)
            W,gradW,subIterW = nnlsm(H.T,A.T,W.T,par.nnls_solver)
            W = W.T
            gradW = gradW.T
            gradH = (W.T @ W) @ H - W.T @ A
        elif par.type == 'regularized':
            H,gradHX,subIterH = nnlsm([W, sbetaI],[A, zerokn],H,par.nnls_solver)
            W,gradW,subIterW = nnlsm([H.T, salphaI],[A.T, zerokm],W.T,par.nnls_solver)
            W, gradW = W.T, gradW.T
            gradH = (W.T @ W) @ H - W.T @ A + par.beta * H
        elif par.type == 'sparse':
            H,gradHX,subIterH = nnlsm([W, sbetaE],[A, zero1n],H,par.nnls_solver)
            W,gradW,subIterW = nnlsm([H.T, salphaI],[A.T, zerokm],W.T,par.nnls_solver)
            W, gradW = W.T, gradW.T
            gradH = (W.T @ W) @ H - W.T @ A + betaI @ H

        if par.verbose:          # collect information for analysis/debugging
            elapsed = cputime - tPrev
            tTotal = tTotal + elapsed
            ver = 0
            idx = iter + 1
#---(1)------(2)--------(3)--------(4)--------(5)---------(6)----------(7)------(8)-----(9)-------(10)--------------(11)-------
# iter # | elapsed | totalTime | subIterW | subIterH | rel. obj.(#) | NM_GRAD | GRAD | DELTA | W density (#) | H density (#)
#------------------------------------------------------------------------------------------------------------------------------
            ver.iter = iter
            ver.elapsed = elapsed
            ver.tTotal = tTotal
            ver.subIterW = subIterW
            ver.subIterH = subIterH
            ver.relError = norm(A-W*H,'fro')/initNorm
            ver.SC1 = getStopCriterion(1,A,W,H,par.type,par.alpha,par.beta,gradW,gradH)/initSCs(1)
            ver.SC2 = getStopCriterion(2,A,W,H,par.type,par.alpha,par.beta,gradW,gradH)/initSCs(2)
            ver.SC3 = getStopCriterion(3,A,W,H,par.type,par.alpha,par.beta,gradW,gradH)/initSCs(3)
            ver.W_density = length(find(W>0))/(m*k)
            ver.H_density = length(find(H>0))/(n*k)
            HIS(idx,1)=iter
            HIS(idx,2)=elapsed
            HIS(idx,3)=tTotal
            HIS(idx,4)=subIterW
            HIS(idx,5)=subIterH
            HIS(idx,6)=ver.relError
            HIS(idx,7)=ver.SC1
            HIS(idx,8)=ver.SC2
            HIS(idx,9)=ver.SC3
            HIS(idx,10)=ver.W_density
            HIS(idx,11)=ver.H_density
            if par.verbose == 2:
                print(ver)
            tPrev = cputime

        if (iter > par.min_iter):
            SC = getStopCriterion(ST_RULE,A,W,H,par.type,par.alpha,par.beta,gradW,gradH)
            if (par.verbose and (tTotal > par.max_time)) or (~par.verbose and ((cputime-tStart) > par.max_time))
                break
            elif (SC / initSC <= par.tol):
                SCconv = SCconv + 1
                if (SCconv >= SC_COUNT):
                    break
            else:
                SCconv = 0
    m,n = A.shape
    norm2=sqrt(sum(np.square(W),axis = 0))
    toNormalize = norm2 > 0
    W[:,toNormalize] = W[:,toNormalize] / np.matlib.repmat(norm2[toNormalize],m,1)
    H[toNormalize,:] = H[toNormalize,:] * np.matlib.repmat(norm2[toNormalize].T,1,n)

    final.iterations = iter
    if par.verbose:
        final.elapsed_total = tTotal
    else:
        final.elapsed_total = cputime - tStart
    final.relative_error = np.linalg.norm(A-W@H, 'fro') / np.linalg.norm(A, 'fro')
    final.W_density = np.count_nonzero(W > 0) / (m * k)
    final.H_density = np.count_nonzero(H > 0) / (n * k)
    print(final)
    return W,H,iter,HIS

#------------------------------------------------------------------------------------------------------------------------
#                                    Utility Functions
#------------------------------------------------------------------------------------------------------------------------
def nnlsm(A,B,init,solver):
    '''solver == "bp" or "as"'''
    return nnlsm_blockpivot(A,B,0,init) if solver == "bp" else nnlsm_activeset(A,B,1,0,init)
#-------------------------------------------------------------------------------
def getInitCriterion(stopRule,A,W,H,type,alpha,beta,gradW = None,gradH = None):
# STOPPING_RULE : 1 - Normalized proj. gradient
#                 2 - Proj. gradient
#                 3 - Delta by H. Kim
#                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if gradW is None or gradH is None:
        gradW,gradH = getGradient(A,W,H,type,alpha,beta)
    m,k = W.shape
    k,n = H.shape
    numAll = m * k + k * n
    if stopRule == 1:
        retVal = np.linalg.norm([gradW, gradH.T], 'fro') / numAll
    elif stopRule == 2:
        retVal = np.linalg.norm([gradW, gradH.T], 'fro')
    elif stopRule == 3:
        retVal = getStopCriterion(3,A,W,H,type,alpha,beta,gradW,gradH)
    elif stopRule == 0:
        retVal = 1
    return retVal
#-------------------------------------------------------------------------------
def getStopCriterion(stopRule,A,W,H,type,alpha,beta,gradW = None,gradH = None):
# STOPPING_RULE : 1 - Normalized proj. gradient
#                 2 - Proj. gradient
#                 3 - Delta by H. Kim
#                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if gradW is None or gradH is None:
        gradW, gradH = getGradient(A,W,H,type,alpha,beta)

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
        resmat=min(H,gradH)
        resvec=resmat[:]
        resmat=min(W,gradW)
        resvec=[resvec, resmat[:]]
        deltao=np.linalg.norm(resvec,1) #L1-norm
        num_notconv=np.count_nonzero(abs(resvec) > 0)
        retVal=deltao / num_notconv
    elif stopRule == 0:
        retVal = 1e100
    return retVal
#-------------------------------------------------------------------------------
def getGradient(A,W,H,type,alpha,beta):
    if type == "plain"
        gradW = W @ (H @ H.T) - A @ H.T
        gradH = (W.T @ W) @ H - W.T @ A
    elif type == 'regularized':
        gradW = W @ (H @ H.T) - A @ H.T + alpha * W
        gradH = (W.T @ W) @ H - W.T @ A + beta * H
    elif type == 'sparse':
        k = W.shape[1]
        betaI = beta * np.ones(k,k)
        gradW = W @ (H @ H.T) - A @ H.T + alpha * W
        gradH = (W.T @ W) @ H - W.T @ A + betaI * H
    return gradW,gradH
