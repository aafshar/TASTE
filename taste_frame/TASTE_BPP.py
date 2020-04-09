import numpy as np
import time

def TASTE_BPP( X,A,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda_):
    #Implementation of PARACouple2
    #


    tStart = time.time()
    RMSE_TIME = []
    ROOTPATH = ''

    J = X[0].shape[1] #  number of features (variables)
    K = max(X.shape)# number of subjects
    Q = []#len(Q) = K

    U = []#len(U) = K
    np.random.seed(seed) # initilizing the modes based on some seed
    V = np.random.rand(J,R)
    W = np.random.rand(K,R)
    H = np.random.rand(R)
    F=np.random.rand(A.shape[1],R)
    for k in range(K):
        U.append(np.random.rand(X[k].shape[0],R))
    prev_RMSE = 0 
    RMSE = 1
    itr = 0
    TOTAL_running_TIME = 0

    beta = 1
    alpha = 1
    while abs(RMSE - prev_RMSE) > conv_tol:
        itr = itr+1
        t_tennn = time.time()
        #update Q_k
        if PARFOR_FLAG:
            for k in range(K):
                T1,_,T2 = np.linalg.svd(mu * (U[k]@H), full_matrices = False)
                Q[k] = T1 @ T2
        else:
            for k in range(K):
                T1,_,T2 = np.linalg.svd(mu * (U[k]@H), full_matrices = False)
                Q[k] = T1 @ T2

        Q_T_U = 0
        if (PARFOR_FLAG):
            for k in range(K):
                Q_T_U += (mu * np.transpose(Q[k]) @ U[k])
        else:
            for k in range(K):
                Q_T_U += (mu * np.transpose(Q[k]) @ U[k])
        H = Q_T_U / (K * mu)


        #update S_k
        V_T_V = np.transpose(V) * V
        F_T_F = np.transpose(F) * F
        if PARFOR_FLAG:
            for k in range(K):
                Khatrio_rao = np.diagflat(np.transpose(U[k]) @ X[k] @ V)
                W[k, :] = np.transpose(nnlsm_blockpivot(((np.transpose(U[k]) @ U[k]) * (V_T_V)) + (lambda_ * F_T_F), Khatrio_rao + (lambda_ * np.transpose(F) @ np.transpose(A[k, :])), 1, np.transpose(W[k, :])))
        else:
            for k in range(K):
                Khatrio_rao = np.diagflat(np.transpose(U[k]) @ X[k] @ V)
                W[k, :] = np.transpose(nnlsm_blockpivot(((np.transpose(U[k]) @ U[k]) * (V_T_V)) + (lambda_ * F_T_F), Khatrio_rao + (lambda_ * np.transpose(F) @ np.transpose(A[k, :])), 1, np.transpose(W[k, :])))
        #update F
        F = np.transpose(nnlsm_blockpivot(lambda_ * np.transpose(W) @ W, lambda_ * np.transpose(W) @ A, 1, np.transpose(F)))

    return TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME,U,Q,H,V,W,F











