import numpy as np
from numpy import linalg as LA

def calculate_RMSE(X,A,U,W,V,F,normX,normA,Size_input,K,PARFOR_FLAG):
    #Calculate fit for parafac2 problem
    RMSE = 0
    fit_tensor = 0
    fit_matrix = 0
    if PARFOR_FLAG:
        #parallel for loop
        for k in range(K):
            M = (U[k]) @ diag(W(k,:)) @ np.transpose(V)
            fit_tensor = fit_tensor +LA.norm(X[k] - M,'fro') ** 2
        RMSE=RMSE+fit_tensor
        fit_tensor=1-(fit_tensor/normX)
    else:
        for k in range(K):
            M = (U[k]) @ diag(W(k,:)) @ np.transpose(V)
            fit_tensor = fit_tensor +LA.norm(X[k] - M,'fro') ** 2
        RMSE=RMSE+fit_tensor
        fit_tensor=1-(fit_tensor/normX)

    RMSE_mat = LA.norm((A - (W @ np.transpose(F)) ),'fro') ** 2
    RMSE=RMSE+RMSE_mat
    RMSE=math.sqrt(RMSE/Size_input)

    fit_matrix=1-(RMSE_mat/normA)

    return fit_tensor, fit_matrix,RMSE

