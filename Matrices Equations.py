import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import newton
from scipy import stats
from IPython.display import display, Math
import cmath
import math
import copy
import random
from random import randint
from numpy.polynomial.hermite import hermval

def print_Matrix(array):
    matrix=''
    for row in array:
        try:
            for number in row:
                matrix+=f'{round(number, 4)}&'
        except TypeError:
            matrix +=f'{row}&'
        matrix=matrix[:-1] + r'\\'
        
    display(Math(r'\begin{bmatrix}' + matrix+r'\end{bmatrix}'))


def norm(H):
    """
    Args:
        H (np.ndarray): symmetric matrix
    
    Returns:
        res**0.5 (float): the norm of the matrix H
    """
    
    line = H.flatten()
    for i in line:
        res += np.abs(i) ** 2
    root = np.sqrt(res)
    
    return root

def off(H):
    
    '''
    Returns the magnitude of the off diagonal elements of matrix A.
    
    Args:
        H (np.ndarray): An n by n matrix.
        
    Variables:
        off (np.ndarray): the flattened matrix A.
        yeeyee (np.ndarray): the returned set of off with the diagonal removed.    
    
    Returns:
        output (int): The square rooted sum of the squares of all matrix A elements.
        
    '''
    off = H.flatten()
    yeeyee = np.delete(off, range(0, len(off), len(H) + 1), 0)
    output = np.sqrt(np.sum(yeeyee * yeeyee))
    
    return output

def jacobi_rotation(A, j, k):
    '''
    Args:
        A (np.ndarray): n by n real symmetric matrix
        j (int): row parameter.
        k (int): column parameter.

    Returns:
        A (np.ndarray): n by n real symmetric matrix, where the A[j,k] and A[k,j] element is zero
        J (np.ndarray): n by n orthogonal matrix, the jacobi_rotation matrix
    '''
        
    if A[j,k] != 0:
        tau = (A[k,k] - A[j,j]) / (2 * A[j,k])
        
        if tau >= 0:
            t = 1 / (tau + np.sqrt(1 + tau ** 2))
        else:
            t = 1 / (tau - np.sqrt(1 + tau ** 2))
            
        c = 1 / np.sqrt(1 + t ** 2)
        s = t * c
    else:
        c = 1
        s = 0
    
    J = np.identity(np.shape(A)[0])
    J[j,j] = c
    J[j,k] = s
    J[k,j] = -s
    J[k,k] = c
    A = np.transpose(J) @ A @ J
    
    return A, J


def real_eigen(A, tol):
    '''
    Args:
        A (np.ndarray): n by n real symmetric matrix
        tol (float): the relative precision
    Returns:
        d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
                        to multiplicity and ordered in non-decreasing order
        R (np.ndarray): n by n orthogonal matrix, R[:,i] is the i-th eigenvector
    '''

    #I couldn't figure out how the psuedocode went wrong so I scrapped it.

    delta = tol * norm(A)
    shape = np.shape(A)[0]
    R = np.identity(shape)
    
    while off(A) > delta:
        copy_matrix = copy.deepcopy(A)
        for i in range(shape):
            copy_matrix[i,i] = 0
        
        j, k = np.where(abs(A) == np.max(abs(copy_matrix)))
        A, J = jacobi_rotation(A, j[0], k[0])
        R = R @ J

    diag = A.diagonal()
    index = np.argsort(diag)
    d = diag[index]
    R = R[:, index]
    
    return d, R


def hermitian_eigensystem(H, tol):
    '''
    Args:
        H (np.ndarray): square complex hermitian matrix
        tol (float): relative precision
    Returns:
        d (np.ndarray): one dimensional array of the ith eigenvalue, repeated according to multiplicity
        Z (np.ndarray): square unitary matrix, U[:,i] is the i-th eigenvector
    '''
        
    # Same here    
        
    shape = np.shape(H)[0]
    S = H.real
    A = H.imag
    
    conc = np.concatenate((np.hstack((S, -A)), np.hstack((A, S))))
    d, R = real_eigen(conc, tol)
    matrix = list(zip(d, R.T))
    matrix.sort(key = lambda x: x[0])
    eigens = np.zeros(shape)
    Z = np.zeros((shape, shape), dtype = 'complex')
    
    for i in range(shape):
        x2 = 2 * i
        eigens[i] = matrix[x2][0]
        Z[:,i] = matrix[x2][1][:shape] + matrix[x2][1][shape:] * 1j 
        
    index = np.argsort(eigens)
    d = eigens[index]
    
    return d, Z


def x_square(N):
    '''
    Models the x squared operator in a shape N by N matrix
    
    Arg:
        N (int): the shape of the matrix to model.
        
    Variables:
        j (int): row parameter.
        k (int): column parameter.
        
    Returns:
        Z (np.ndarray): An N by N matrix of the x squared operator.
    '''
    
    Z = np.zeros((N, N))
    
    for j in range(N):
        for k in range(N):
            if j == k:
                Z[j, k] = j + 0.5
            elif i == j + 2:
                Z[j, k] = 0.5 * ((j - k) * j)**(1./2.)
            elif i == j - 2:
                Z[j, k] = 0.5 * ((j + 1) * (j + 2))**(1./2.)
                
    return Z

def x_quad(N):
    '''
    Models the x**4 operator in a shape N by N matrix
    
    Arg:
        N (int): the shape of the matrix to model.
        
    Variables:
        j (int): row parameter.
        k (int): column parameter.
        
    Returns:
        Z (np.ndarray): An N by N matrix of the x**4 operator.
    '''
    
    Z = np.zeros((N, N))
    
    for j in range(N):
        for k in range(N):
            if j == k:
                Z[j, k] = (1 / 4) * (6 * j**2 + 6 * j + 3)
            elif j == k + 2:
                Z[j, k] = (j - 0.5) * ((j - 1) * j) ** (1 / 2)
            elif j == k - 2:
                Z[j, k] = (j + 1.5) * ((j + 1) * (j + 2)) ** (1./2.)
            elif j == k + 4:
                Z[j, k] = (1 / 4) * ((j - 1) * (j - 2) * (j - 3) * j) ** (1 / 2)
            elif j == k - 4:
                Z[j, k] = (1 / 4) * ((j + 1) * (j + 2) * (j + 3) * (j + 4)) ** (1 / 2)
        
    return Z

def anharmonic_oscillator(lambduh, N):
    '''
    The modeling of an anharmonic oscillator.
    
    Arg:
        lambduh (int): the lambda value of the energy of the system.
        N (int): The size of matrix to consider.
        
    Variables:
        I (np.ndarray): an N by N size identity matrix.
        aplus (np.ndarray): an N by N size model of the raising operator.
        
    Returns:
        H (np.ndarray): Anharmonic Oscillator matrix.
    '''
    I = np.identity(N)
    aplus = np.zeros((N, N))
    
    for i in range(1, N):
        aplus[i, i - 1] = i ** (1 / 2)
    
    H = lambduh * (1 + x_quad(N)) + np.dot(aplus, aplus.T) + (1 / 2) * I
        
    return H


def eigenfunction(col, num, lambduh = 0):
    '''
    Args:
        col (int): Set of eigenvectors
        num (int): range of positions
        
    Kwargs:
        lambduh (int): 

    Returns:
        reals (int):
    '''
    
    m_size = 20
    M = anharmonic_oscillator(lambduh, m_size)
    U = hermitian_eigensystem(M, 1e-10)[1]
    norm = U[:, col]
    sums = 0
    zeroes = np.zeros(m_size, dtype = 'complex')
    
    for j in range(m_size):
        zeroes[j] = norm[j]
        sums += ((2 ** j * math.factorial(j) * np.pi ** 0.5) ** (-0.5)) * np.exp(-0.5 * num ** 2) * hermval(num, zeroes)
        
    reals = sums.real
    
    return reals
