import numpy as np

def claculate_norm(X,A,K,PARFOR_FLAG):
    #UNTITLED3 Summary of this function goes here
    #   Detailed explanation goes here
    normX = 0
    Size_input=(A.shape[0] * A.shape[1])
    num_non_z = np.count_nonzero(A)
    normA = np.sum(np.square(A))
    if PARFOR_FLAG:
        #parallel for loop
        for k in range(K):
            normX += np.sum(np.square(X[k]))
            Size_input += (X[k].shape[0] * X[k].shape[1])
            num_non_z += np.count_nonzero(X[k])
    else:
        for k in range(K):
            normX += np.sum(np.square(X[k]))
            Size_input += (X[k].shape[0] * X[k].shape[1])
            num_non_z += np.count_nonzero(X[k])

    return normX,normA,Size_input

