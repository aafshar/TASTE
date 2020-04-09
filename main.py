from taste_frame import *
from nonnegfac import *
import numpy as np
# this package is from https://www.cc.gatech.edu/~hpark/nmfsoftware.php



##create a synthetic data set.
K = 100
J = 40#number of dynamic features
P = 30#number of static features
I_k=50
R=4  #number of factors or components
H=np.random.rand(R,R)
W=np.random.rand(K,R)
F=np.random.rand(P,R)
V=np.random.rand(J,R)
Q=[]#len(Q) = K
U=[]#len(U) = K

for k in range(K):
	Q.append(np.zeros([I_k, R]))
    col_Q_k = np.random.randint(5, size=(I_k, 1)) + 1
    Temp_Q = np.matlib.repmat(col_Q_k, 1, R)
    for r in range(R):
        Q[k][:, r] = Temp_Q[:,r] == r
    Q[k] = Q[k] / np.sqrt(np.sum(np.square(Q[k]), axis=0))

    U[k] = Q[k] @ H

A = W @ np.transpose(F)
X = [] #len(X) = K
for i in range(K):
    X.append((U[i] @ np.diagflat(W[i, :])) @ np.transpose(V))




data_name = "Synethetic_data"

lambda_ = 1
mu = 1
conv_tol = 1e-5 #converegance tolerance
PARFOR_FLAG = 0 #parallel computing
normX, normA, Size_input = claculate_norm(X,A,K,PARFOR_FLAG) #Calculate the norm of the input X
Constraints = ['nonnegative', 'nonnegative','nonnegative','nonnegative']

itr = 5
seed = 1

TOTAL_running_TIME,rmse,FIT_Tensor,FIT_Matrix,RMSE_TIME,U,Q,H,V,W,F = TASTE_BPP(X,A,R,conv_tol,seed,PARFOR_FLAG,normX,normA,Size_input,Constraints,mu,lambda_)
plot(RMSE_TIME(:,1),RMSE_TIME(:,2))
xlabel("Time")
ylabel("RMSE")

