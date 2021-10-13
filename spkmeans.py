from enum import Enum
import numpy as np
from numpy.linalg import norm
from os.path import isfile, splitext
import sys
import pandas as pd
import random
from spkmeansmodule import *

ERR_INPUT = "Invalid Input!"
ERR_GENERAL = "An Error Has Occured"


class Goal(Enum):
    spk = 0,
    wam = 1,
    ddg = 2,
    lnorm = 3,
    jacobi = 4


CENTROIDS = []
N = 0 # number of points
K = 0 # number of centroids
D = 0
MAX_ITER = 300


def parse_cmdline_args():
    global N, K, D, datapoints, goal
    '''gets data from user and formats it correctly'''
    args = iter(sys.argv[1:])  # forget about arg[0] (module name)
    argc = len(sys.argv[1:])
    assert 3 == argc, ERR_INPUT
    try: K = int(next(args))
    except ValueError: raise AssertionError(ERR_INPUT)

    try:
        goal = {'spk' : Goal.spk,
                'wam' : Goal.wam,
                'ddg' : Goal.ddg,
                'lnorm' : Goal.lnorm,
                'jacobi' : Goal.jacobi}[next(args)]
    except KeyError:
        raise AssertionError(ERR_INPUT)

    filename = next(args)
    validate_filename(filename)
    df = pd.read_csv(filename, header=None)
    datapoints = df.to_numpy()
    N = len(datapoints)
    D = len(datapoints[0])
    assert K<N, ERR_INPUT



def validate_filename(filename):
    assert isfile(filename), f"{filename}: No such file"
    _, file_ext = splitext(filename)
    assert file_ext == ".txt" or file_ext == ".csv", ERR_INPUT


def calc_cur_d(centroids_indexes, datapoints, z):
    '''calculates the array as given in the assignment'''
    D = np.array([
        min([norm(np.subtract(datapoints[i], datapoints[centroids_indexes[j]]))**2 for j in range(z)])
        for i in range(N)
    ])
    return D

def calc_p(D_values):
    '''calculates P(x_i) like the definition given in the homework'''
    s = np.sum(D_values)
    return np.array([D_values[i] / s for i in range(N)])

def kmeans_pp(datapoints):
    np.random.seed(0)
    centroid_indexes = [np.random.choice(range(N))]
    for z in range (1, K):
        D_values = calc_cur_d(centroid_indexes, datapoints, z)
        P = calc_p(D_values)
        centroid_indexes.append(np.random.choice(range(N), p = P)) #chose centroid by probabilities according to P
    return centroid_indexes

def print_list(lst, fmt=False):
    for item in lst[:-1]:
        if -5.0e-5 < item < 5e-5: item = 0
        if fmt: print(f"{item:.4f}".rstrip('0'), end=",")
        else: print(f"{item}", end=",")
    item = lst[-1]
    if -5.0e-5 < item < 5e-5: item = 0
    if fmt: print(f"{item:.4f}".rstrip('0'), end="")
    else: print(f"{item}", end="")

def main():
    '''gets data from user and returns start centroids, points, N, K, MAX_ITER'''
    np.random.seed(0)
    global CENTROIDS, datapoints, K, MAX_ITER, N, goal
    try:
        parse_cmdline_args()
    except AssertionError as e:
        print(e)
        exit(1)

    if goal == Goal.wam:
        wam(datapoints, K, N, D)
    elif goal == Goal.ddg:
        ddg(datapoints, K, N, D)
    elif goal == Goal.lnorm:
        lnorm(datapoints, K, N, D)
    elif goal == Goal.jacobi:
        jacobi(datapoints, K, N, D)
    elif goal == Goal.spk:
        Tmat = spk_getT(datapoints, K, N, D)
        K = len(Tmat[0])
        centroid_indexes = kmeans_pp(Tmat)
        centroid_list = [list(Tmat[i]) for i in centroid_indexes]
        print_list(centroid_indexes)
        print()
        spk(Tmat, centroid_list, K, N, D)


if __name__ == "__main__":
    main()

