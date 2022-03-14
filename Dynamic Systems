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

# Dynamic Systems

def dynamics_solve(f, D = 1, t_0 = 0.0, s_0 = 1, h = 0.1, N = 100, method = "Euler"):
    
    """ Solves for dynamics of a given dynamical system
    
    - User must specify dimension D of phase space.
    - Includes Euler, RK2, RK4 modeling, that user can choose from using the keyword "method"
    
    Args:
        f: A python function f(t, s) that assigns a float to each time and state representing
        the time derivative of the state at that time.
        
    Kwargs:
        D: Phase space dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        s_0: Initial state (float for D=1, ndarray for D>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4"
    
    Returns:
        T: Numpy array of times
        S: Numpy array of states at the times given in T
    """
    
    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if D == 1:
        S = np.zeros(N + 1)
    
    if D > 1:
        S = np.zeros((N + 1, D))
        
    S[0] = s_0
    
    if method == 'Euler':
        for n in range(N):
            S[n + 1] = S[n] + h * f(T[n], S[n])

    if method == 'RK2':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + (2./3.) * h, S[n] + (2./3.) * k1)
            S[n + 1] = S[n] + 0.25 * k1 + 0.75 * k2
    
    if method == 'RK4':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + 0.5 * h, S[n] + 0.5 * k1)
            k3 = h * f(T[n] + 0.5 * h, S[n] + 0.5 * k2)
            k4 = h * f(T[n] + h, S[n] + k3 )
            S[n + 1] = S[n] + (1./6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return T, S



def hamiltonian_solve(d_qH, d_pH, d = 1, t_0 = 0.0, q_0 = 0.0, p_0 = 1.0, h = 0.1, N = 100, method = "Euler"):
    
#     Q0 = K*(1-e)
#     P0 = np.sqrt(mu*(1+e)/(1-e))

    
    """ Solves for dynamics of Hamiltonian system
    
    - User must specify dimension d of configuration space.
    - Includes Euler, RK2, RK4, Symplectic Euler (SE) and Stormer Verlet (SV) 
      that user can choose from using the keyword "method"
    
    Args:
        d_qH: Partial derivative of the Hamiltonian with respect to coordinates (float for d=1, ndarray for d>1)
        d_pH: Partial derivative of the Hamiltonian with respect to momenta (float for d=1, ndarray for d>1)
        
    Kwargs:
        d: Spatial dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        q_0: Initial position (float for d=1, ndarray for d>1) set to 0.0 as default
        p_0: Initial momentum (float for d=1, ndarray for d>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4", "SE", "SV"
    
    Returns:
        T: Numpy array of times
        Q: Numpy array of positions at the times given in T
        P: Numpy array of momenta at the times given in T
    """
    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if d == 1:
        P = np.zeros(N + 1)
        Q = np.zeros(N + 1)
        
        Q[0] = q_0
        P[0] = p_0
    
    if d > 1:
        P = np.zeros((N + 1, d))
        Q = np.zeros((N + 1, d))
        
        P[0][0] = p_0[0]
        P[0][1] = p_0[1]
        Q[0][0] = q_0[0]
        Q[0][1] = q_0[1]
    
    if method == 'Euler':
        for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n], P[n])
    
    if method == 'RK2':
        for n in range(N):
            k1_Q = h * d_pH(Q[n], P[n])
            k1_P = h * (- d_qH(Q[n], P[n]))
            
            
            k2_Q = h * d_pH(Q[n] + (2./3.) * h, P[n] + (2./3.) * k1_Q)
            k2_P = h * -d_qH(Q[n] + (2./3.) * h, P[n] + (2./3.) * k1_P)
            
            Q[n + 1] = Q[n] + 0.25 * k1_Q + 0.75 * k2_Q
            P[n + 1] = P[n] + 0.25 * k1_P + 0.75 * k2_P
            
    if method == 'RK4':
        for n in range(N): 
            k1_Q = h * d_pH(Q[n], P[n])
            k1_P = h * (- d_qH(Q[n], P[n]))
            
            k2_Q = h * d_pH(Q[n] + 0.5 * h, P[n] + 0.5 * k1_Q)
            k2_P = h * -d_qH(Q[n] + 0.5 * h, P[n] + 0.5 * k1_P)
            
            k3_Q = h * d_pH(Q[n] + 0.5 * h, P[n] + 0.5 * k2_Q)
            k3_P = h * -d_qH(Q[n] + 0.5 * h, P[n] + 0.5 * k2_P)
            
            k4_Q = h * d_pH(Q[n] + h, P[n] + k3_Q)
            k4_P = h * -d_qH(Q[n] + h, P[n] + k3_P)
            
            Q[n + 1] = Q[n] + (1./6.) * (k1_Q + 2 * k2_Q + 2 * k3_Q + k4_Q)
            P[n + 1] = P[n] + (1./6.) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)
        
    if method == 'SE':
        for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n+1], P[n])
    
    if method == "SV" and d > 1:
        for n in range(N):
            Pnhalf0 = P[n][0] - d_qH(Q[n][0], P[n][0])[0]
            Pnhalf1 = P[n][1] - d_qH(Q[n][1], P[n][1])[1]
            
            Q[n + 1][0] = Q[n][0] + d_pH(Q[n][0], Pnhalf0)[0]
            Q[n + 1][1] = Q[n][1] + d_pH(Q[n][1], Pnhalf1)[1]
            
            P[n + 1][0] = Pnhalf0 - d_qH(Q[n + 1][0], P[n][0])[0]
            P[n + 1][1] = Pnhalf1 - d_qH(Q[n + 1][1], P[n][1])[1]
    
    if method == "SV" and d == 1:
        for n in range(N):
            Pnhalf = P[n] - h/2 * d_qH(Q[n], P[n])
            Q[n + 1] = Q[n] + h * d_pH(Q[n], Pnhalf)
            P[n + 1] = Pnhalf - h/2 * d_qH(Q[n+1], P[n])
        
        
    return T, Q, P


def deriv_qh(q, p):
        
    """ The derivative of the Hamiltonian of a system with respect to position.
        In this circumstance, a harmonic oscillator.
    
    Args:
        q: Position (float)
        p: Momentum (float)
        
    Variables:
        m: Mass (float)
        w: Angular frequency (float)
    
    Returns:
        ret: The derivative of the hamiltonian with respect to position (float)
    """
    
    m = 0.5
    w = 1
    ret = m * w **2 * q
    return ret

def deriv_ph(q, p):
    
    """ The derivative of the Hamiltonian of a system with respect to momentum.
        In this circumstance, a harmonic oscillator.
    
    Args:
        q: Position (float)
        p: Momentum (float)
        
    Variables:
        m: Mass (float)
    
    Returns:
        ret: The derivative of the hamiltonian with respect to momentum (float)
    """
    m = 0.5
    ret = 1/m * p
    return ret



def pop_sol(p_0, t, R = 0.2, K = 1e6):
            
    """ Models a population growth over a period of time.
    
    
    Args:
        p_0: Initial Population (float)
        t: Time (float)
    
    Kwargs:
        R: Coefficient characterizing growth rate. (float)
        K: Carrying capacity of the environment. (float)
    
    Returns:
        p: Population at the end of an elapsed time. (float)
    """
    
    p = (K * p_0) / ((K - p_0) * np.exp(-R * t) + 1)
    return p


def basic_pop_model(t, P, R = 0.2, K = 1e6):
                
    """ Models a population growth over a period of time.
    
    
    Args:
        P: Initial Population (float)
        t: Time (float)
    
    Kwargs:
        R: Coefficient characterizing growth rate. (float)
        K: Carrying capacity of the environment. (float)
    
    Returns:
        population : Population at the end of an elapsed time. (float)
    """
    
    population = R * (1 - P / K) * P
    return population


def C_pop_model(T, P):
                    
    """ Models a population growth over a period of time.
    
    
    Args:
        P: Initial Population (float)
        T: Time (float)
    
    Variables:
        R: Coefficient characterizing growth rate. (float)
        K: Carrying capacity of the environment. (float)
        C: Constant (float)
        Pc: Constant (float)
    
    Returns:
        f : Population at the end of an elapsed time. (float)
    """
    
    R = 0.2
    K = 1000
    C = 40
    Pc = 100
    f = R * (1 - P/K) * P - C * (P ** 2 / (Pc ** 2 + P **2))
    return f
