import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
# this package is from https://www.cc.gatech.edu/~hpark/nmfsoftware.php
import importlib
import sys

if "." not in sys.path:
    sys.path.append(".")

import taste_frame
import nonnegfac

# importlib.reload(taste_frame)
# importlib.reload(nonnegfac)

def A_join_X(A, X_df):
    X_height = X_df.shape[0]
    K = A.shape[0]
    X = []
    j = 0
    for i in range(K):
        while j < X_height and A.iloc[i, 0] != X_df.iloc[j, 0]:
            j += 1
        start = j
        while j < X_height and A.iloc[i, 0] == X_df.iloc[j, 0]:
            j += 1
        temp = X_df.iloc[start : j, 1:]
        X.append(csr_matrix(([1 for _ in range(temp.shape[0])], ((temp["r"]-1), (temp["code"]-1))), shape = (temp["r"].iloc[-1], max(X_df["code"]))))
    return X

def my_plot(RMSE_TIME, name):
    fig = plt.figure()
    plt.plot([tup[0] for tup in RMSE_TIME], [tup[1] for tup in RMSE_TIME])
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.savefig(name)

def main(R, static, dynamic, use_saved_np):
    if use_saved_np:
        with np.load('AX.npz', allow_pickle = True) as data:
            X_case = data['X_case']
            X_ctrl = data['X_ctrl']
            A_case = data['A_case']
            A_ctrl = data['A_ctrl']
    else:
#         A_df = pd.read_csv(static, header = 0)
        A_df = pd.read_csv(static,  names=["patient_id","sex","race_white","race_black","race_others","race_hispanic","esrd","sp_alzhdmta","sp_chf","sp_chrnkidn","sp_cncr","sp_copd","sp_depressn","sp_ischmcht","sp_osteoprs","sp_ra_oa","sp_strketia","leq68","leq74","leq82","geq82","is_case"])
        A_case = A_df[A_df["is_case"] == 1]
        A_ctrl = A_df[A_df["is_case"] == 0]

        X_df = pd.read_csv(dynamic, names=["patient_id", "r", "code"])

        X_case = A_join_X(A_case, X_df)
        A_case = A_case.iloc[:, 1:-1].to_numpy() # A_case = np.ones(12494, 1)
        X_ctrl = A_join_X(A_ctrl, X_df)
        A_ctrl = A_ctrl.iloc[:, 1:-1].to_numpy() # A_ctrl = np.ones(12494, 1)
        np.savez_compressed("AX.npz", X_case = X_case, A_case = A_case, X_ctrl = X_ctrl, A_ctrl = A_ctrl)

    lambda_ = 1
    mu = 1
    conv_tol = 1e-4 #converegance tolerance
    PARFOR_FLAG = 0 #parallel computing
    Constraints = ['nonnegative', 'nonnegative','nonnegative','nonnegative']
    seed = 1

    normX, normA, Size_input = taste_frame.claculate_norm(X_case,A_case,A_case.shape[0],PARFOR_FLAG) #Calculate the norm of the input X_case
    TOTAL_running_TIME,rmse,FIT_Tensor,FIT_Matrix,RMSE_TIME_case,U_case,Q_case,H_case,V_case,W_case,F_case = taste_frame.TASTE_BPP(X_case,A_case,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda_)
    my_plot(RMSE_TIME_case, str(R) + ".png")

    normX, normA, Size_input = taste_frame.claculate_norm(X_ctrl,A_ctrl,A_ctrl.shape[0],PARFOR_FLAG) #Calculate the norm of the input X_ctrl
    TOTAL_running_TIME,RMSE,FIT_T,FIT_M,RMSE_TIME_ctrl,U_ctrl,Q_ctrl,H_ctrl,V_ctrl,W_ctrl,F_ctrl = taste_frame.PARACoupl2_BPP( X_ctrl,A_ctrl,V_case,F_case,H_case,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda_ )
    my_plot(RMSE_TIME_ctrl, str(R) + "_projection.png")

    return RMSE_TIME_case,U_case,Q_case,H_case,V_case,W_case,F_case,RMSE_TIME_ctrl,U_ctrl,Q_ctrl,H_ctrl,V_ctrl,W_ctrl,F_ctrl

if __name__ == '__main__':
    main(R = 5, static = "data/static.csv", dynamic = "data/dynamic.csv", use_saved_np = False)
