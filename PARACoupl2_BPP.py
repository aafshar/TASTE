import time
import numpy as np
import importlib
import taste_frame
import nonnegfac
importlib.reload(taste_frame)
importlib.reload(nonnegfac)

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
        FIT_T, FIT_M,RMSE = taste_frame.calculate_RMSE( X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG )
        RMSE_TIME.append((TOTAL_running_TIME, RMSE))

    return TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME,U,Q,H,V,W,F
