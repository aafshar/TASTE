import numpy as np

def solveNormalEqComb( AtA,AtB,PassSet = None):
# Solve normal equations using combinatorial grouping.
# Although this function was originally adopted from the code of
# "M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450",
# important modifications were made to fix bugs.
#
# Modified by Jingu Kim (jingu@cc.gatech.edu)
#             School of Computational Science and Engineering,
#             Georgia Institute of Technology
#
# Last updated Aug-12-2009

    iter = 0
    if PassSet is None or np.count_nonzero(PassSet) == 0 or np.count_nonzero(PassSet == 0) == 0:
        Z = np.linalg.solve(AtA, AtB)
        iter = iter + 1
    else:
        Z = np.zeros(AtB.shape)
        n, k1 = PassSet.shape

        ## Fixed on Aug-12-2009
        if k1 == 1:
            Z[PassSet]=np.linalg.solve(AtA[PassSet, PassSet], AtB[PassSet])
        else:
            ## Fixed on Aug-12-2009
            # The following bug was identified by investigating a bug report by Hanseung Lee.
            sortedPassSet = sorted(enumerate(PassSet.T.tolist()), key = lambda x: x[1])
            sortIx = [x[0] for x in sortedPassSet]
            sortedPassSet = [x[1] for x in sortedPassSet]
            breakIx = [0, np.where(np.diff(sortedPassSet, axis = 0).T.any(axis = 0))[0], k1]

            for k in range(len(breakIx)-1):
                cols = sortIx[breakIx[k]+1 : breakIx[k+1]]
                vars = PassSet[:, sortIx[breakIx[k] + 1]]
                Z[vars, cols] = np.linalg.solve(AtA[vars,vars], AtB[vars,cols])
                iter += 1
    return Z, iter
