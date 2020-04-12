import numpy as np

# ------------------------------ Reconstruction test ------------------------------
m = 300
n = 200
k = 10

W_org = np.random.rand(m,k)
W_org[np.random.rand(m,k) > 0.5] = 0
H_org = np.random.rand(k,n)
H_org[np.random.rand(k,n) > 0.5] = 0

# normalize W, since 'nmf' normalizes W before return
norm2 = math.sqrt(sum(np.square(W_org), axis = 0))
toNormalize = norm2 > 0
W_org[:,toNormalize] = np.linalg.solve(np.matlib.repmat(norm2[toNormalize],m,1), W_org[:,toNormalize])

A = W_org @ H_org

W,H,iter,HIS = nmf(A,k,'type','plain','tol',1e-4)

# -------------------- column reordering before computing difference
reorder = np.zeros(k,1)
selected = np.zeros(k,1)
for i in range(k):
    for j in range(k):
        if ~selected[j]:
            break
    minIx = j

    for j in range(minIx, k):
        if ~selected[j]:
            d1 = np.linalg.norm(W[:,i]-W_org[:,minIx])
            d2 = np.linalg.norm(W[:,i]-W_org[:,j])
            if d2 < d1:
                minIx = j
    reorder[i] = minIx
    selected[minIx] = 1

W_org = W_org[:,reorder]
H_org = H_org[reorder,:]
# ---------------------------------------------------------------------

recovery_error_W = np.linalg.norm(W_org-W)/np.linalg.norm(W_org)
recovery_error_H = np.linalg.norm(H_org-H)/np.linalg.norm(H_org)
