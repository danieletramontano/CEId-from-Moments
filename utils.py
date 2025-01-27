"""
This module contains the utility functions.
"""
from itertools import permutations
import numpy as np

def reduce_list(l):
    """
    Return the element of the list if it has only one element.
    """
    if len(l) == 1:
        return l[0]
    return l

def min_dist_perm(v1, v2):
    """
    Find the permutation of v2 that minimizes the L1 distance to v1.
    """
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("Dimensions do not match")
    perms = list(permutations(range(v1.shape[0])))
    match = perms[0]
    min_dist = np.sum(np.abs((v1 - v2)))

    for i in perms[1:]:
        dist = np.sum(np.abs((v1 - v2[list(i)])))
        if dist < min_dist:
            min_dist = dist
            match = i
    match = np.array(match)

    return match

def project(x, a, c):
    """
    Project x onto the hyperplane defined by a and c.

    Parameters:
    x (numpy.ndarray): The point to project.
    a (numpy.ndarray): The normal vector of the hyperplane.
    c (float): The offset of the hyperplane.
    """
    return x-a*(np.dot(x, a) - c)/np.dot(a, a)


def get_ratio_(Z, D, deg=2):
    """
    Computes ratio between alpha_d/alpha_z

    Parameters:
        - Z (np.array): proxy variable observations
        - D (np.array): treatment observations
        - deg (int): moment of non-guassianity (equal to the (n-1) from the original paper)
    """
    var_u = np.mean(Z*D)
    sign = np.sign(var_u)

    diff_normal_D = np.mean(D**(deg)*Z) - deg*var_u*np.mean(D**(deg-1))
    diff_normal_Z = np.mean(Z**(deg)*D) - deg*var_u*np.mean(Z**(deg-1))

    alpha_sq = ((diff_normal_D) / (diff_normal_Z))
    if alpha_sq < 0:
        alpha_sq = -(abs(alpha_sq)**(1/(deg-1)))
    else:
        alpha_sq = alpha_sq**(1/(deg-1))
    alpha_sq = abs(alpha_sq)*sign

    return alpha_sq


def cross_moment(Z, D, Y, deg=2):
    """
    Cross-Moment method implementation

    Parameters:
        - Z (np.array): proxy variable observations
        - D (np.array): treatment observations
        - Y (np.array): outcome observations
        - deg (int): moment of non-guassianity (equal to the (n-1) from the original paper)
    """

    denominator = 0
    while denominator==0:
        alpha_sq = get_ratio_(Z, D, deg)
        numerator = np.mean(D*Y) - alpha_sq*np.mean(Y*Z)
        denominator = np.mean(D*D) - alpha_sq*np.mean(D*Z)
        deg += 1
    return numerator / denominator