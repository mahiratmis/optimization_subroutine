

"""#This version is same as earlier version _v3

# There are 4 possible variants of _vs
# Initial population: Restricted, Unrestricted
# Mutation: Switch only, Switch and Open new Cluster

####This version Unrestricted + Switch Only ######

# For all variants we use following GA parameters:
# Population: 100
# Generation:25
# Crossover:0.8
# Mutation:0.4
# Gene Mutation:0.4 """
from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
cimport cython
#import math
from libc cimport math 
import json
import time
import random
from deap import base
from deap import creator
from deap import tools
import sys
import os
import csv

import operator


import concurrent.futures as cf
import multiprocessing
# sys.path.append(os.getcwd())

# import sframe as sf
# from bokeh.charts import BoxPlot, output_notebook, show
# import seaborn as sns
# import pulp
# from pulp import *

# Import DEAP from and documentation
# https://deap.readthedocs.io/en/master/
# pip install deap


# Below functions are needed for Queuing approximation
# Define the generator that will generate all possible assignments of classes
# to servers, without permutations.


def generateVectorsFixedSum(int m, int n):
    """ generator for all combinations of $w$ for given number of servers and
     classes """
    cdef int i
    cdef list vect 
    if m == 1:
        yield [n]
    else:
        for i in range(n + 1):
            for vect in generateVectorsFixedSum(m - 1, n - i):
                yield [i] + vect


# @DecorateProfiler
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef MMCsolver(np.ndarray lamda, np.ndarray mu, int nservers, int mClasses):
    # print(f"lambda {type(lamda)}, mu {type(mu)}, nservers{type(nservers)}, mclass{type(mClasses)}")
    cdef int i, i1, n, j1, j2, q_max, q_max_nplus, q_max_n, q_max_nminus, m,q 
    cdef double eps, lambdaTot, delta, EQTotal, EQQmin1Total, EQ2Total 
    cdef np.ndarray alpha, A0, A1, I, Z, Z_prev, A0_inv, A1_n, A0_n, L_n, inv1minZ, EQ, EQ2, EN, ES2, ESq, EN2, VarN, marginalN, inv1minAlphaZ
    cdef list Q, idxMat, i_map_full, P
    # assert sum(lamda/mu) < nservers  # ensure stability
    # initialize \Lamda and \alpha
    lambdaTot = sum(lamda)
    alpha = lamda/lambdaTot
    # print(f"lambdaTot {type(lambdaTot)}, alpha {type(alpha)}")
    # create mapping between the combination vectors and matrix columns/rows
    idx_map = dict([(tuple(vect), i)
                    for i,
                    vect in enumerate(generateVectorsFixedSum(mClasses, nservers))])
    # need to use tuple here as 'list' cannot be as a key
    i_map = dict([(idx_map[idx], list(idx)) for idx in idx_map])
    # need to use list here as 'tuple' cannot be modified as will be need further

    # generate matrices A_0 and A_1
    q_max = len(idx_map)
    A0 = np.zeros((q_max, q_max))  # corresponds to terms with i items in queue
    A1 = np.zeros((q_max, q_max))  # corresponds to terms with i+1 items in queue
    for i, idx in i_map.items():
        # diagonal term
        A0[i, i] += 1 + np.sum(idx*mu)/lambdaTot

    # term corresponding to end of service for item j1, start of service for j2
        for j1 in xrange(mClasses):
            for j2 in xrange(mClasses):
                idx[j1] += 1
                idx[j2] -= 1
                i1 = idx_map.get(tuple(idx), -1)  # convert 'list' back to tuple to use it as a key
                if i1 >= 0:
                    A1[i, i1] += alpha[j2]/lambdaTot*idx[j1]*mu[j1]
                idx[j1] -= 1
                idx[j2] += 1

    # compute matrix Z iteratively
    eps = 0.00000001
    # print(f"eps {type(eps)}")
    I = np.eye(q_max)  # produces identity matrix
    Z_prev = np.zeros((q_max, q_max))
    delta = 1
    A0_inv = np.linalg.inv(A0)
    while delta > eps:
        Z = np.dot(A0_inv, I + np.dot(A1, np.dot(Z_prev, Z_prev)))  # invA0*(I+A1*Z*Z)
        delta = np.sum(np.abs(Z-Z_prev))
        # print(f"delta {type(delta)}")
        Z_prev = Z

    # generate Q matrices, it will be stored in a list
    Q = []
    idxMat = []  # matrix with server occupancy for each system state, will be used in computing the system parameters
    Q.insert(0, Z[:])
    idxMat.insert(0, np.array([x for x in i_map.values()]))

    i_map_full = []
    i_map_full.append(i_map)

    # dict([ (tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, nServers)) ])
    idx_map_nplus = idx_map
    i_map_nplus = i_map  # dict([(idx_map_nplus[idx], list(idx)) for idx in idx_map_nplus ])
    q_max_nplus = len(idx_map_nplus)

    idx_map_n = idx_map_nplus
    i_map_n = i_map_nplus
    q_max_n = q_max_nplus

    A1_n = A1[:]

    for n in range(nservers, 0, -1):
        idx_map_nminus = dict([(tuple(vect), i)
                               for i, vect in enumerate(generateVectorsFixedSum(mClasses, n-1))])
        i_map_nminus = dict([(idx_map_nminus[idx], list(idx)) for idx in idx_map_nminus])
        q_max_nminus = len(idx_map_nminus)

        i_map_full.insert(0, i_map_nminus)

        L_n = np.zeros((q_max_n, q_max_nminus))  # corresponds to terms with i items in queue
        A0_n = np.zeros((q_max_n, q_max_n))  # corresponds to terms with i items in queue
        for i, idx in i_map_n.items():

            # diagonal term
            A0_n[i, i] += 1 + np.sum(idx*mu)/lambdaTot

            # term corresponding to arrival of item item j1
            for j2 in xrange(mClasses):
                idx[j2] -= 1
                i2 = idx_map_nminus.get(tuple(idx), -1)
                if i2 >= 0:
                    L_n[i, i2] += alpha[j2]
                idx[j2] += 1

        # Q_n = (A_0 - A_1*Q_{n+1})^{-1}*L_n
        Q.insert(0, np.dot(np.linalg.inv(A0_n-np.dot(A1_n, Q[0])), L_n))

        idx_map_nplus = idx_map_n
        # i_map_nplus = i_map_n
        q_max_nplus = q_max_n

        idx_map_n = idx_map_nminus
        i_map_n = i_map_nminus
        q_max_n = q_max_nminus
        idxMat.insert(0, np.array([x for x in i_map_n.values()]))

        A1_n = np.zeros((q_max_n, q_max_nplus))  # corresponds to terms with i+1 items in queue
        for i, idx in i_map_n.items():
            # term corresponding to end of service for item j1
            for j1 in xrange(mClasses):
                idx[j1] += 1
                i1 = idx_map_nplus.get(tuple(idx), -1)
                if i1 >= 0:
                    A1_n[i, i1] += idx[j1]*mu[j1]/lambdaTot
                idx[j1] -= 1

    # compute the P_n for n<k and normalize it such that sum(P_n) = 1
    P = []
    P.append([1.0])

    sm = 1.0
    for n in xrange(nservers):
        P.append(np.dot(Q[n], P[-1]))
        sm += sum(P[-1])

    sm += sum(np.dot(np.linalg.inv(np.eye(len(P[-1])) - Z), np.dot(Z, P[-1])))
    # print(f"sm {type(sm)}")
    for p in P:
        p[:] /= sm  # normalization

    # compute totals needed for the E[Q_i] - marginal distributions
    inv1minZ = np.linalg.inv(np.eye(len(P[-1])) - Z)
    EQTotal = sum(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), Z), P[-1]))
    EQQmin1Total = 2 * \
        sum(np.dot(np.dot(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), inv1minZ), Z), Z), P[-1]))
    EQ2Total = EQQmin1Total + EQTotal

    # compute 1st and 2nd marginal moments of the numbers in the queue E[Q_i] and E[Q_i^2]
    EQ = alpha*EQTotal
    EQQmin1 = alpha*alpha*EQQmin1Total
    EQ2 = EQQmin1 + EQ

    # compute 1st and 2nd marginal moments of the numbers in the system E[N_i] and E[N_i^2]
    ENTotal = EQTotal + sum(lamda/mu)
    EN = EQ + lamda/mu

    # TODO compute the E[N_i^2]
    ES2 = np.zeros(mClasses)
    for (p, idx) in zip(P[:-1], idxMat[:-1]):
        ES2 += np.dot(p, idx**2)
    ES2 += np.dot(np.dot(inv1minZ, P[-1]), idxMat[-1]**2)

    ESq = alpha*np.dot(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), Z), P[-1]), idxMat[-1])

    EN2 = EQ2 + 2*ESq + ES2

    # compute marginal variances of the numbers in the queue Var[Q_i] and in the system Var[N_i]
    # VarQTotal = EQ2Total - EQTotal**2
    # VarQ = EQ2 - EQ**2

    VarN = EN2 - EN**2

    # computeMarginalDistributions
    qmax = 1500
    # print(f"qmax {type(qmax)}")
    marginalN = np.zeros((mClasses, qmax))

    for m in xrange(mClasses):
        for imap, p in zip(i_map_full[:-1], P[:-1]):
            for i, idx in imap.items():
                marginalN[m, idx[m]] += p[i]

        inv1minAlphaZ = np.linalg.inv(np.eye(len(P[-1])) - (1-alpha[m])*Z)
        frac = np.dot(alpha[m]*Z, inv1minAlphaZ)
        # tmp = np.dot(self.Z, self.P[-1])
        # tmp = np.dot(inv1minAlphaZ, tmp)
        tmp = np.dot(inv1minAlphaZ, P[-1])

        for q in xrange(0, qmax):
            for i, idx in i_map_full[-1].items():
                if idx[m]+q < qmax:
                    marginalN[m, idx[m]+q] += tmp[i]
            tmp = np.dot(frac, tmp)
    # exit(0)
    return marginalN, EN, VarN

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline (double, double, double) whittApprox(double E1, double E2, double E3) :
    '''
    input: first 3 moments of hyperexpo dist.
    returns: parameters of hyperexpo (p, v1 and v2)
    uses whitt approximation.....
    '''
    cdef double x,y,Ev1,Ev2,p

    x = E1*E3-1.5*E2**2
    # print x
    # assert x >= 0.0

    y = E2-2*(E1**2)
    # print y
    # assert y >= 0.0

    Ev1 = ((x+1.5*y**2+3*E1**2*y)+math.sqrt((x+1.5*y**2-3*E1**2*y)**2+18*(E1**2)*(y**3)))/(6*E1*y)
    # print Ev1
    # assert Ev1 >= 0

    Ev2 = ((x+1.5*y**2+3*E1**2*y)-math.sqrt((x+1.5*y**2-3*E1**2*y)**2+18*(E1**2)*(y**3)))/(6*E1*y)
    # assert Ev2 >= 0

    p = (E1-Ev2)/(Ev1-Ev2)
    # assert p >= 0

    return 1.0/Ev1, 1.0/Ev2, p

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bint isServiceRateEqual(list mu) :
    return len(set(mu)) <= 1


def Approx_MMCsolver(lamda, mu, nServers, mClasses):
    '''
    inputs: lamda->failure rates of SKUs
            mu ->service rates of servers for SKUs
            nServers->number of servers in repairshop
            mClasses->number of SKUs := length of failure rates

    output: Marginal Queue length for each type of SKU
            Expected Queue length  ''  ''   ''   ''
            Variance of Queue length ''  '' ''   ''

    solution: Approximate 3 class system and calls MMCsolver
    '''
    print("Being Usedddddddddddddddd")
    marginalN = []
    EN = []
    VarN = []

    for mCl in range(mClasses):
        # first moment for service time distribution for approximation:
        E_S1 = (np.inner(lamda, 1/mu)-(lamda[mCl]*1/mu[mCl]))/(sum(lamda)-lamda[mCl])  # checked

        # second moment
        E_S2 = 2*(np.inner(lamda, (1/mu)**2) -
                  (lamda[mCl]*(1/mu[mCl])**2))/(sum(lamda)-lamda[mCl])  # checked

        # third moment
        E_S3 = 6*(np.inner(lamda, (1/mu)**3) -
                  (lamda[mCl]*(1/mu[mCl])**3))/(sum(lamda)-lamda[mCl])  # checked

        # calculate inputs for to check neccesity condtion:
        varA = E_S2-E_S1**2
        cA = math.sqrt(varA)/E_S1

        # to check if all of the service rates of approximated service are same
        # if it is true sum of hyperexpo with same parameter is ---> exponential distribution

        mu_copy = []
        mu_copy[:] = mu
        del mu_copy[mCl]

        if isServiceRateEqual(mu_copy) is True:
            # we can assume there is only aggreate remaing streams to one rather than two
            p = 1
            v1 = mu_copy[0]

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            # if sum(lamda/mu)>nservers:
            #    nservers
            # we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1]), np.array([mu[mCl], v1]), nservers=nServers, mClasses=2)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

        # if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        # E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)

        else:
            # a2 calculation
            a2 = (6*E_S1-(3*E_S2/E_S1))/((6*E_S2**2/4*E_S1)-E_S3)

            # a1 calculation
            a1 = (1/E_S1)+(a2*E_S2/(2*E_S1))

            # v1 calculation
            v1 = (1.0/2.0)*(a1+math.sqrt(a1**2-4*a2))

            # v2 calculation
            v2 = (1.0/2.0)*(a1-math.sqrt(a1**2-4*a2))

            # p calculation
            p = 1-((v2*(E_S1*v1-1))/float((v1-v2)))

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            lamA2 = (1-p)*(sum(lamda)-lamda[mCl])
            # SA2=1/float(v2)

            # Now we have 3 classes of streams (2 streams for approximation) as usual
            # so mClasses=3

            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1, lamA2]), np.array([mu[mCl], v1, v2]), nservers=nServers, mClasses=3)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

    return marginalN, EN, VarN

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef Approx_MMCsolver2(np.ndarray lamda, np.ndarray mu, int nservers, int mClasses) :
    '''
    inputs: lamda->failure rates of SKUs
            mu ->service rates of servers for SKUs
            nservers->number of servers in repairshop
            mClasses->number of SKUs := length of failure rates

    output: Marginal Queue length for each type of SKU
            Expected Queue length  ''  ''   ''   ''
            Variance of Queue length ''  '' ''   ''

    solution: Approximate 3 class system and calls MMCsolver
    '''

    # print nservers
    cdef list marginalN = [], EN = [], VarN = [], mu_copy
    cdef int mCl
    cdef double E_S1, E_S2, E_S3, varA, cA, lam1, lamA1, lamA2, v1, v2, p, tot_temp

    for mCl in range(mClasses):
        # first moment for service time distribution for approximation:
        tot_temp = sum(lamda)-lamda[mCl]
        E_S1 = (np.inner(lamda, 1/mu)-(lamda[mCl]*1/mu[mCl]))/tot_temp  # checked
        # print E_S1
        # second moment
        E_S2 = 2*(np.inner(lamda, (1/mu)**2) -
                  (lamda[mCl]*(1/mu[mCl])**2))/tot_temp # checked

        # third moment
        E_S3 = 6*(np.inner(lamda, (1/mu)**3) -
                  (lamda[mCl]*(1/mu[mCl])**3))/tot_temp  # checked

        # calculate inputs for to check neccesity condtion:
        varA = E_S2-E_S1**2
        cA = math.sqrt(varA)/E_S1

        # print(f"E_s1 {type(E_S1)} E_s2 {type(E_S2)} E_s3 {type(E_S3)} varA {type(varA)} cA {type(cA)}")

        assert (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3) > 0

        # to check if all of the service rates of approximated service are same
        # if it is true sum of hyperexpo with same parameter is ---> exponential distribution

        mu_copy = []
        mu_copy[:] = mu
        del mu_copy[mCl]

        if isServiceRateEqual(mu_copy) is True:
            # we can assume there is only aggreate remaing streams to one rather than two
            p = 1
            v1 = mu_copy[0]

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # print(f"lam1 {type(lam1)} lamA1 {type(lamA1)}")
            # SA1=1/float(v1)

            # sum(lamda/mu)<nservers
            tot_temp = lam1/mu[mCl] + lamA1/v1
            if tot_temp > nservers:
                # print "hasan"
                nservers = int(tot_temp)+1

            # we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1]), np.array([mu[mCl], v1]), nservers, mClasses=2)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])
            # print "aaaa"

        # if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        # E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)

        else:

            v1, v2, p = whittApprox(E_S1, E_S2, E_S3)
            # print v1
            # print v2

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            lamA2 = (1-p)*(sum(lamda)-lamda[mCl])
            # SA2=1/float(v2)
            # print(f"lam1 {type(lam1)} lamA1 {type(lamA1)} lamA2 {type(lamA2)}")
            tot_temp = lam1/mu[mCl] + lamA1/v1 + lamA2/v2
            if tot_temp >= nservers:
                # print "turan"
                nservers = int(tot_temp)+1
            # Now we have 3 classes of streams (2 streams for approximation) as usual
            # so mClasses=3

            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1, lamA2]), np.array([mu[mCl], v1, v2]), nservers, mClasses=3)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

    return marginalN, EN, VarN, nservers


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef parallel_sum_over_columns(double[:, :] matrix):
    start_time = time.time()
    cdef int m = matrix.shape[0]
    cdef int n = matrix.shape[1]
    cdef int i, j
    cdef double accumulator=0.0
    cdef double[:] result = np.zeros(m, dtype=np.double)

    with nogil, parallel(num_threads=8):
        for i in prange(m, schedule='guided'):
            accumulator = 0.0
            for j in range(n):
                accumulator = accumulator + matrix[i, j]
            result[i] = accumulator

    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef parallel_sum(double[:] vect):
    start_time = time.time()
    cdef Py_ssize_t i, n = len(vect)
    cdef double accumulator = 0.0
    for i in prange(n, nogil=True):
        accumulator += vect[i]
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return accumulator
# code for optimization inventories after given queue length distribution
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef OptimizeStockLevelsAndCosts(np.ndarray holdingCosts,  double penalty, np.ndarray marginalDistribution):
    cdef int sk, nSKUs, maxQueue
    cdef double hCost, pCost, totalCost
    cdef np.ndarray S, PBO, EBO

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if not marginalDistribution.shape[1]:
        marginalDistribution = marginalDistribution.reshape(1, len(marginalDistribution))

    nSKUs = len(holdingCosts)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.sum(marginalDistribution[:, 1:], axis=1)
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts/penalty
    for sk in xrange(nSKUs):
        while S[sk] < maxQueue and np.sum(marginalDistribution[sk, S[sk]+1:]) > hb_ratio[sk]:
            S[sk] += 1
            # -= marginalDistribution[sk, S[sk]]
            PBO[sk] = np.sum(marginalDistribution[sk, S[sk]+1:])
            EBO[sk] = np.sum(marginalDistribution[sk, S[sk]:]*n_array[:-S[sk]])  # -= PBO[sk]

    totalCost = np.sum(S*holdingCosts) + np.sum(penalty*EBO)
    hCost = np.sum(S*holdingCosts)
    pCost = np.sum(penalty*EBO)
    # print ((EBO < 0).sum() == EBO.size).astype(np.int)
    # if pCost<0.0:
    #    print EBO
    # print  ((EBO < 0).sum() == EBO.size).astype(np.int)
    # print all(i >= 0.0 for i in marginalDistribution)

    return totalCost, hCost, pCost, S, EBO


@cython.boundscheck(False)
@cython.nonecheck(False)
cdef individual2cluster(list individual):
    '''
    -input: list of integers representing assingment of SKUs to clusters
    -output: list of list representing clusters and assinged SKUs in each cluster
    '''
    cdef int i,j,x
    return [[i + 1 for i, j in enumerate(individual) if j == x] for x in set(individual)]


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef evalOneMax(np.ndarray FailureRates, 
    np.ndarray ServiceRates, 
    np.ndarray holding_costs,
    double penalty_cost, 
    double skillCost, 
    double machineCost, 
    list individual):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation

           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)

     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of
             servers at each cluster

     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repetead
           (5) Step (4) is repated if there is a decrease in total cost


    Warning !! type matching array vs list might be problem (be careful about type matching)

    '''

    # from individual to cluster
    cdef list cluster_GA = individual2cluster(individual)
    # bestCost=float('inf')
    # bestCluster=[]
    # print "\n"
    # print individual
    # print cluster_GA

    cdef list bestS = [], bestEBO = [], EBO_cluster = [], S_cluster = []
    cdef list bestserverAssignment = [], serverAssignment = [], sliceIndex2 = []
    cdef double TotalCost = 0.0, totalCostClust, hCost, pCost, temp_hCost, temp_pCost, TotalMachine_Cost, TotalSkill_Cost, temp_totalCostClust, temp_TotalMachine_Cost, temp_TotalSkill_Cost
    cdef double TotalHolding=0.0, TotalPenalty=0.0, TotalSkillCost=0.0, TotalMachineCost = 0.0, temp_total=0.0
    cdef int i = 0, len_slice = 0, min_nserver=0, min_nserverUpdate, x
    cdef np.ndarray sRate, fRate, hcost
    # LogFileList=[]
    # logFile={}
    # iterationNum=0
    for cluster in cluster_GA:
        sliceIndex2[:] = cluster
        sliceIndex2[:] = [x - 1 for x in sliceIndex2]
        # print(sliceIndex2)
        len_slice = len(sliceIndex2)
        sRate = np.array(ServiceRates[sliceIndex2], dtype=np.double )
        fRate = np.array(FailureRates[sliceIndex2], dtype=np.double)
        hcost = np.array(holding_costs[sliceIndex2], dtype=np.double)
        min_nserver = int(sum(fRate/sRate))+1
        # print sliceIndex2
        # print "RUn FINISHED \n"
        # sys.exit(0)
        # costTemp=0
        # while costTemp<=machineCost:
        if len_slice<= 3:
            marginalDist, _, _ = MMCsolver(fRate, sRate, min_nserver, len_slice)
        else:
            marginalDist, _, _, min_nserverUpdate = Approx_MMCsolver2(
                fRate, sRate, min_nserver, len_slice)
            min_nserver = min_nserverUpdate

        totalCostClust, hCost, pCost, S, EBO = OptimizeStockLevelsAndCosts(
            hcost, penalty_cost, np.array(marginalDist))

        # increasing number of servers and checking if total cost decreases

        TotalMachine_Cost = min_nserver*machineCost
        TotalSkill_Cost = min_nserver*len_slice*skillCost

        totalCostClust = totalCostClust+TotalMachine_Cost+TotalSkill_Cost

        while True:
            min_nserver += 1
            if len_slice<= 3:
                marginalDist, _, _ = MMCsolver(fRate, sRate, min_nserver, len_slice)
            else:
                marginalDist, _, _, min_nserverUpdate = Approx_MMCsolver2(
                    fRate, sRate, min_nserver, len_slice)
                min_nserver = min_nserverUpdate

            temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO = OptimizeStockLevelsAndCosts(
                hcost, penalty_cost, np.array(marginalDist))
            temp_TotalMachine_Cost = min_nserver*machineCost
            temp_TotalSkill_Cost = min_nserver*len_slice*skillCost

            temp_totalCostClust = temp_totalCostClust+temp_TotalMachine_Cost+temp_TotalSkill_Cost

            if temp_totalCostClust > totalCostClust:
                min_nserver -= 1
                break
            else:
                totalCostClust = temp_totalCostClust

                TotalMachine_Cost = temp_TotalMachine_Cost
                TotalSkill_Cost = temp_TotalSkill_Cost
                hCost = temp_hCost
                pCost = temp_pCost

        TotalHolding += hCost
        TotalPenalty += pCost

        TotalSkillCost += TotalSkill_Cost
        TotalMachineCost += TotalMachine_Cost

        TotalCost = TotalCost+totalCostClust

        EBO_cluster.append(EBO.tolist())
        S_cluster.append(S.tolist())
        serverAssignment.append(min_nserver)

    return TotalCost,

# bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, \
#            bestserverAssignment, LogFileList
# DONT FORGET COME AT THE END!!!


def Final_evalOneMax(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, individual):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation

           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)

     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of
             servers at each cluster

     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repeted
           (5) Step (4) is repated if there is a decrease in total cost


    Warning !! type matching array vs list might be problem (be careful about type matching)

    '''

    # from individual to cluster
    cluster_GA = individual2cluster(individual)
    # bestCost=float('inf')
    # bestCluster=[]
    bestS = []
    bestEBO = []
    EBO_cluster = []
    S_cluster = []
    bestserverAssignment = []
    serverAssignment = []
    sliceIndex2 = []
    TotalCost = 0.0
    TotalHolding, TotalPenalty, TotalSkillCost, TotalMachineCost = 0.0, 0.0, 0.0, 0.0
    # LogFileList=[]
    # logFile={}
    # iterationNum=0
    for cluster in cluster_GA:
        sliceIndex2[:] = cluster
        sliceIndex2[:] = [x - 1 for x in sliceIndex2]

        sRate = np.array(ServiceRates[sliceIndex2])
        fRate = np.array(FailureRates[sliceIndex2])
        hcost = np.array(holding_costs[sliceIndex2])

        min_nserver = int(sum(fRate/sRate))+1

        # costTemp=0
        # while costTemp<=machineCost:
        if len(sRate) <= 3:
            marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
        else:
            marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(
                fRate, sRate, min_nserver, len(fRate))
            min_nserver = min_nserverUpdate

        totalCostClust, hCost, pCost, S, EBO = OptimizeStockLevelsAndCosts(
            hcost, penalty_cost, np.array(marginalDist))

        # increasing number of servers and checking if total cost decreases

        TotalMachine_Cost = min_nserver*machineCost
        TotalSkill_Cost = min_nserver*len(fRate)*skillCost

        totalCostClust = totalCostClust+TotalMachine_Cost+TotalSkill_Cost

        while True:
            min_nserver += 1
            if len(sRate) <= 3:
                marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
            else:
                marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(
                    fRate, sRate, min_nserver, len(fRate))
                min_nserver = min_nserverUpdate

            temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO = OptimizeStockLevelsAndCosts(
                hcost, penalty_cost, np.array(marginalDist))
            temp_TotalMachine_Cost = min_nserver*machineCost
            temp_TotalSkill_Cost = min_nserver*len(fRate)*skillCost

            temp_totalCostClust = temp_totalCostClust+temp_TotalMachine_Cost+temp_TotalSkill_Cost

            if temp_totalCostClust > totalCostClust:
                min_nserver -= 1
                break
            else:
                totalCostClust = temp_totalCostClust

                TotalMachine_Cost = temp_TotalMachine_Cost
                TotalSkill_Cost = temp_TotalSkill_Cost
                hCost = temp_hCost
                pCost = temp_pCost

        TotalHolding += hCost
        TotalPenalty += pCost

        TotalSkillCost += TotalSkill_Cost
        TotalMachineCost += TotalMachine_Cost

        TotalCost = TotalCost+totalCostClust

        EBO_cluster.append(EBO.tolist())
        S_cluster.append(S.tolist())
        serverAssignment.append(min_nserver)

    return TotalCost, TotalHolding, TotalPenalty, TotalMachineCost, TotalSkillCost, cluster_GA, S_cluster, EBO_cluster, serverAssignment
# DONT FORGET COME AT THE END!!!


def swicthtoOtherMutation(individual, indpb):
    '''
    input- individual chromosome
    output- some genes changed to other genes in chromosome (changing clusters)
    There might be other ways of mutation - swaping clusters of two SKUs (crossover does that) two way swap
                                          - opening a new cluster
                                          - closing a cluster and allocated SKUs in that cluster to another cluster
                                          -(local or tabu search idea!!)
    '''
    # to keep orginal probabilty of switching to other cluster during iteration
    individual_copy = individual[:]
    for i in range(len(individual)):
        if random.random() <= indpb:
            if random.random() <= 1.5:  # switch only version _v4a
                # set is used to give equal probability to assign any other cluster
                # without set there is a higher probablity to assigning to a cluster that inclludes more SKUs
                if len(list(set(individual_copy).difference(set([individual_copy[i]])))) >= 1:
                    individual[i] = random.choice(
                        list(set(individual_copy).difference(set([individual_copy[i]]))))

            else:
                # This mutation type aimed for generating new cluster and going beyond the allowed maximum num cluster
                if len(list(set(range(1, len(individual_copy)+1)).difference(set(individual_copy)))) >= 1:
                    individual[i] = random.choice(
                        list(set(range(1, len(individual_copy)+1)).difference(set(individual_copy))))

    return individual
