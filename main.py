from taste_frame import *
from nonnegfac import *
import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
# this package is from https://www.cc.gatech.edu/~hpark/nmfsoftware.php



A = readtable('static_feature.csv')
XK = readtable('design_matrix_3.csv')
[K, P] = size(A)
X_height = height(XK)
R = max(XK{:, 3})
X = cell(K, 1)
j = 1
for k = 1:K
    start = j
    while (j <= X_height) && strcmp(A{k, 1}{1}, XK{j, 1}{1})
        j = j + 1
    end
    X{k} = XK{start : (j-1), 2:(end-1)}
end
A = A(:, 2:(end-1))
for i = 1:K
    A{i, 2} = 2010-(A{i, 2} - mod(A{i, 2}, 10000))/10000
end
A = A{:, :}
for k = 1:K
    hei = size(X{k})
    X{k} = sparse(X{k}(:, 1), X{k}(:, 2), ones(hei(1), 1), X{k}(end, 1), R)
end

R = 5


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
plot(RMSE_TIME[:,0],RMSE_TIME[:,1])
xlabel("Time")
ylabel("RMSE")

