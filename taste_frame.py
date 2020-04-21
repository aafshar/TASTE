import math

import numpy as np
from numpy import linalg as LA
import numpy as np
import scipy
from scipy.sparse import *
from scipy.sparse.linalg import norm
import time
import nonnegfac
import importlib
importlib.reload(nonnegfac)


def claculate_norm(X, A, K, PARFOR_FLAG):
    # UNTITLED3 Summary of this function goes here
    #   Detailed explanation goes here
    normX = 0
    Size_input = A.shape[0] * A.shape[1]
    num_non_z = np.count_nonzero(A)
    normA = np.sum(np.square(A))
    if PARFOR_FLAG:
        # parallel for loop
        for k in range(K):
            normX += scipy.sparse.linalg.norm(X[k], 'fro') ** 2
            Size_input += X[k].shape[0] * X[k].shape[1]
            num_non_z += X[k].getnnz()
    else:
        for k in range(K):
            normX += scipy.sparse.linalg.norm(X[k], 'fro') ** 2
            Size_input += (X[k].shape[0] * X[k].shape[1])
            num_non_z += X[k].getnnz()

    return normX, normA, Size_input


def calculate_RMSE(X, A, U, W, V, F, normX, normA, Size_input, K, PARFOR_FLAG):
    # Calculate fit for parafac2 problem
    RMSE = 0
    fit_tensor = 0
    fit_matrix = 0
    # if PARFOR_FLAG:
    for k in range(K):
        M = U[k] @ np.diag(W[k, :]) @ V.T
        fit_tensor = fit_tensor + LA.norm(X[k] - M, 'fro') ** 2
    RMSE = RMSE + fit_tensor
    fit_tensor = 1 - (fit_tensor / normX)

    RMSE_mat = LA.norm((A - (W @ F.T)), 'fro') ** 2
    RMSE = RMSE + RMSE_mat
    RMSE = math.sqrt(RMSE / Size_input)

    fit_matrix = 1 - (RMSE_mat / normA)

    return fit_tensor, fit_matrix, RMSE


def TASTE_BPP(X, A, R, conv_tol, seed, PARFOR_FLAG, normX, normA, Size_input, Constraints, mu, lambda_):

    tStart = time.time()
    RMSE_TIME = []
    ROOTPATH = ''
    J = X[0].shape[1]  # number of features (variables)
    K = len(X)  # number of subjects
    Q = []  # len(Q) = K

    U = []  # len(U) = K
    np.random.seed(seed)  # initilizing the modes based on some seed
    V = np.random.rand(J, R)
    W = np.random.rand(K, R)
    H = np.random.rand(R)
    F = np.random.rand(A.shape[1], R)
    for k in range(K):
        U.append(np.random.rand(X[k].shape[0], R))
    prev_RMSE = 0
    RMSE = 1
    itr = 0
    TOTAL_running_TIME = 0

    beta = 1
    alpha = 1
    while abs(RMSE - prev_RMSE) > conv_tol:
        itr = itr + 1
        t_tennn = time.time()
        # update Q_k
        # if PARFOR_FLAG:
        for k in range(K):
            T1, _, T2 = np.linalg.svd(mu * (U[k] @ H.reshape(-1, 1)), full_matrices=False)
            Q.append(T1 @ T2)

        Q_T_U = 0
        if (PARFOR_FLAG):
            for k in range(K):
                Q_T_U += (mu * np.transpose(Q[k]) @ U[k])
        else:
            for k in range(K):
                Q_T_U += (mu * np.transpose(Q[k]) @ U[k])
        H = Q_T_U / (K * mu)

        # update S_k
        V_T_V = V.T @ V
        f_t_f = F.T @ F
        # if PARFOR_FLAG:
        for k in range(K):
            k_hatrio_rao = np.diag(U[k].T @ X[k] @ V)
            W[k, :] = nonnegfac.nnlsm_blockpivot(((U[k].T @ U[k]) * V_T_V) + (lambda_ * f_t_f),
                                                 (k_hatrio_rao + lambda_ * F.T @ A[k, :].T).reshape(-1, 1), 1, W[k, :].T)[0].T
        # update F
        F = nonnegfac.nnlsm_blockpivot(lambda_ * W.T @ W, lambda_ * W.T @ A, 1, F.T)[0].T

        U_S_T_U_S = 0
        U_S_T_X = 0
        # update V
        if PARFOR_FLAG:
            for k in range(K):
                U_S = U[k] * W[k, :]  # element wise multiplication
                U_S_T_U_S = U_S_T_U_S + np.transpose(U_S) @ U_S
                U_S_T_X += np.transpose(U_S) @ X[k]
        else:
            for k in range(K):
                U_S = U[k] * W[k, :]  # element wise multiplication
                U_S_T_U_S = U_S_T_U_S + np.transpose(U_S) @ U_S
                U_S_T_X += np.transpose(U_S) @ X[k]
        V = nonnegfac.nnlsm_blockpivot(U_S_T_U_S, U_S_T_X, 1, V.T)[0].T

        # if PARFOR_FLAG:
        for k in range(K):
            V_S = V * W[k, :]  # element wise multiplication
            V_S_T_V_S = np.transpose(V_S) @ V_S + mu * np.eye(R)
            U_S_T_X = np.transpose(V_S) @ np.transpose(X[k]) + (mu * np.transpose(H) @ np.transpose(Q[k]))
            U[k] = nonnegfac.nnlsm_blockpivot(V_S_T_V_S, U_S_T_X, 1, np.transpose(U[k]))[0].T

        tEnd = time.time()
        TOTAL_running_TIME = TOTAL_running_TIME + (tEnd - tStart)
        prev_RMSE = RMSE
        FIT_T, FIT_M, RMSE = calculate_RMSE(X, A, U, W, V, F, normX, normA, Size_input, K, PARFOR_FLAG)

        RMSE_TIME.append((TOTAL_running_TIME, RMSE))

    return TOTAL_running_TIME, RMSE, FIT_T, FIT_M, RMSE_TIME, U, Q, H, V, W, F

def PARACoupl2_BPP( X,A,V,F,H,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda_ ):

    tStart=time.time()
    RMSE_TIME=[]
    ROOTPATH = ''

    J=X[0].shape[1] #  number of features (variables)
    K = len(X)# number of subjects
    Q = []
    U = []

    np.random.seed(seed)  # initilizing the modes based on some seed
    W = np.random.rand(K,R)
    for k in range(K):
        U.append(np.random.rand(X[k].shape[0],R))
    prev_RMSE=0
    RMSE=1
    itr=0
    TOTAL_running_TIME=0

    beta=1
    alpha=1
    while abs(RMSE - prev_RMSE) > conv_tol:
        itr = itr + 1
        t_tennn = time.time()
        # update Q_k
        # if PARFOR_FLAG:
        for k in range(K):
            T1, _, T2 = np.linalg.svd(mu * (U[k] @ H.reshape(-1, 1)), full_matrices=False)
            Q.append(T1 @ T2)

        #update S_k
        V_T_V=V.T @ V
        F_T_F=F.T @ F
        # if (PARFOR_FLAG)
        for k in range(K):
            k_hatrio_rao = np.diag(U[k].T @ X[k] @ V)
            W[k, :] = nonnegfac.nnlsm_blockpivot(((U[k].T @ U[k]) * V_T_V) + (lambda_ * F_T_F),
                                                 (k_hatrio_rao + lambda_ * F.T @ A[k, :].T).reshape(-1, 1), 1, W[k, :].T)[0].T
        #update U_k

        # if PARFOR_FLAG:
        for k in range(K):
            V_S = V * W[k, :]  # element wise multiplication
            V_S_T_V_S = V_S.T @ V_S + mu * np.eye(R)
            # V_S_T_V_S=sparse(V_S_T_V_S)
            U_S_T_X = V_S.T @ X[k].T + (mu * H.T @ Q[k].T)
            # U_S_T_X=sparse(U_S_T_X)
            U[k] = nonnegfac.nnlsm_blockpivot(V_S_T_V_S, U_S_T_X, 1, U[k].T)[0].T

        tEnd = time.time()
        TOTAL_running_TIME = TOTAL_running_TIME + (tEnd - tStart)
        prev_RMSE = RMSE
        FIT_T, FIT_M,RMSE = calculate_RMSE( X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG )
        RMSE_TIME.append((TOTAL_running_TIME, RMSE))

    return TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME,U,Q,H,V,W,F