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


# Markov chain Monte Carlo simulation

def weighted_coin(beta, n, steps, plotshow = False):
    '''
    Get the average earnings of a weighted coin using a MCMC simulation, where you are heads over a number of trials.
    
    Args:
    beta (int): probability of getting a head as opposed to a tails
    n (int): total number of iterations
    steps (int): steps size for averaging
    
    KwArgs:
    plotshow (bool): deciding if you want to plot or not.
    
    Returns:
    earning_avg (int): average earnings from the weighted coin.
    '''
    coin = ['H', 'T']
    prob = [beta, 1 - beta]
    heads = 0
    tails = 0
    avg_earn = []
    iters = 0
    state = coin[random.randint(0,1)]
    
    for i in range(0, n, steps):
        for i in range(steps):
            if state == "H":
                randumb = np.random.rand(1)

                if randumb < 0.5:
                    heads += 1
                    state = "H"
                else:
                    pacc = min(1, prob[1]/prob[0])
                    randummy = np.random.rand(1)
                    
                    if (1 - pacc) > randummy:
                        heads += 1
                        state = "H"
                    else:
                        state = "T"
                        tails += 1

            else:
                randumb = np.random.rand(1)

                if randumb > 0.5:
                    heads += 1
                    state = "H"
                else:
                    state = "T"
                    tails += 1
    
        iters += steps
        heads_percent = heads / iters
        tails_percent = 1 - heads_percent
        
        earnings = 2 * heads_percent - 1
        avg_earn.append(earnings)
        
        earning_avg = avg_earn[-1]
        
    if plotshow:
        print("Average Earnings: ", avg_earn[-1])

        plt.plot(range(len(avg_earn)), avg_earn)
        plt.xlabel("Flips (in thousands)")
        plt.ylabel("Probability of Heads")
        plt.title("Probability of Getting Heads")
        plt.grid()
        plt.show()
    
    return earning_avg


def average_earnings_per_flip(beta):
    '''
    Get the theoretical average earnings per flip.
    
    Arg:
    beta (int): probability of getting a heads over a tails.
    
    Returns:
    results (int): Earnings expected per flip.
    '''
    results = 2 * beta - 1
    return results


def gen_grid_two_plots():
    '''
    plotting theoretical earnings for a weighted coin using a MCMC simulation vs a theoretical weighted coin.
    '''
    fig = plt.figure(figsize = (10, 10))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    
    twenty = np.linspace(0.001, 1, 20)
    theo = []
    avg_earn1 = []
    avg_earn2 = []
    
    
    for num in twenty:
        theo.append(average_earnings_per_flip(num))
        avg_earn1.append(weighted_coin(num, 1000, 50))
        avg_earn2.append(weighted_coin(num, 1000000, 20000))
    
    labels = ["Weighted Coin", "Theoretical"]
    ax1.plot(twenty, avg_earn1, label = labels[0])
    ax1.plot(twenty, theo, label = labels[1])
    ax1.set_xlabel("Beta")
    ax1.set_ylabel("Average Earnings ($)")
    ax1.set_title(f"Average Earnings vs. Beta")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(twenty, avg_earn2, label = labels[0])
    ax2.plot(twenty, theo, label = labels[1])
    ax2.set_xlabel("Beta")
    ax2.set_ylabel("Average Earnings ($)")
    ax2.set_title(f"Average Earnings vs. Beta")
    ax2.legend()
    ax2.grid()
    
    plt.show()


def weighted_die(n, plotshow = False):
    '''
    Getting the theoretical earnings from a weighted die using an MCMC simulation. This is intending you weighted 1 and 2 as
    three times more likely than 3, 4, 5, and 6.
    
    Args:
    n (int): number of simulations
    
    KwArgs:
    plotshow (bool): decides whether you want to show the graph or not.
    
    Returns:
    earnings (int): average earnings from the weighted die
    '''
    die = ['low', 'high']
    prob = [.6, .4]
    wins = 0
    losses = 0
    avg_earn = []
    state = die[random.randint(0,1)]
    
    for i in range(0, n):
        if state == "low":
            randumb = np.random.rand(1)

            if randumb < 0.5:
                wins += 1
                state = "low"
            else:
                pacc = min(1, prob[1]/prob[0])
                randummy = np.random.rand(1)

                if (1 - pacc) > randummy:
                    wins += 1
                    state = "low"
                else:
                    state = "high"
                    losses += 1

        else:
            randumb = np.random.rand(1)

            if randumb > 0.5:
                wins += 1
                state = "low"
            else:
                state = "high"
                losses += 1
    
    win_percentage = wins / n
    earnings = 2 * win_percentage - 1
        
    if plotshow:
        print("Average Earnings: ", avg_earn[-1])

        plt.plot(range(len(avg_earn)), avg_earn)
        plt.xlabel("Rolls (in thousands)")
        plt.ylabel("Probability of Winning")
        plt.title("Probability of Winning a Weighted Die")
        plt.grid()
        plt.show()
    
    return earnings


def two_dim_ising(L, temp, num_steps):
    '''
    Two dimensional structure of spins. A MCMC simulation is run to test average energy and magnetization.
    
    Arguments:
    L (int): Lattice of side length (number of length)
    temp (int): given temperature
    num_steps (int): number of updates to make in the MCMC update.
    
    Returns:
    t (int list): list of the number of each steps at each interval for U and M added.
    U (int list): average internal energy of the system at each step interval
    M (int list): average magnetization of the system at each step interval
    '''
    
    S = 0
    E = 0
    E_avg = []
    S_avg = []
    steps = [] 
    
    spins_list = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            spins_list[i][j] = np.random.choice([-1, 1])
            S += spins_list[i][j]
            
    for i in range(L):
        for j in range(L): # -= because of the negative sum
            E -= spins_list[i][j] * (spins_list[(i + 1) % L][j] + spins_list[(i - 1) % L][j]  + spins_list[i][(j + 1) % L] + spins_list[i][(j - 1) % L])
            
    E_avg.append(E)
    S_avg.append(S)
    steps.append(0)
        
    for i in range(num_steps):
        for j in range(150):
            row = np.random.randint(L)
            col = np.random.randint(L)
            spot = spins_list[row][col]
            
            if spot == 1:
                dS = -2
            else:
                dS = 2
            
            dE = 2 * spot * (spins_list[(row + 1) % L][col] + spins_list[(row - 1) % L][col]  + spins_list[row][(col + 1) % L] + spins_list[row][(col - 1) % L])
            
            if dE < 0:
                spins_list[row][col] += dS 
                E += dE
                S += dS
            else:
                if np.random.rand(1) < np.exp(-dE / temp):
                    spins_list[row][col] += dS
                    E += dE
                    S += dS
        
        iplus1 = E_avg[i] + 1/(i+1) * (E - E_avg[i])
        E_avg.append(iplus1) 
        Siplus1 = S_avg[i] + 1/(i+1) * (S - S_avg[i])
        S_avg.append(Siplus1) 
        steps.append((i + 1) * 100)
        

    t = [(L ** -2) * i for i in steps]
    U = [(L ** -2) * i for i in E_avg]
    M = [(L ** -2) * i for i in S_avg]
    
    
    
    return t, U, M


def M_theoretical(T):
    '''
    Theoretical magnetization of a system based on a critical temperature of the system.
    
    Args:
    T (int): temperature of system
    
    Returns:
    Ms (int list): list of theoretical magnetizations
    '''
    
    Tc = 2 / np.log(1 + np.sqrt(2))
    Mt = []
    for i in T:
        if i < Tc:
            answer = (1 - np.sinh(2 / i)**-4) ** (1/8)
        else:
            answer = 0
        Mt.append(answer)
        
    return Mt



def two_dim_ising_temp(spins_list, E, S, L, temp, num_steps):
    '''
    Two dimensional structure of spins. A MCMC simulation is run to test average energy and magnetization at different temperatures
    
    Args:
    spins_list (int numpy ndarray): array of spins values (1 or -1) replicating a lattice structure
    E (int): initial energy
    S (int): initial net spin
    L (int): size of lattice side
    temp (int): temperature of system
    num_steps (int): number of steps in MCMC update
    
    Returns:
    spins_list (int numpy ndarray): array of spins values (1 or -1) replicating a lattice structure, but updated with a change
    E (int): final energy after change
    S (int): final net spin after change
    M (int): magnetization
    '''    
    
    S_avg = [S]
            
    for i in range(num_steps):
        for j in range(150):
            row = np.random.randint(L)
            col = np.random.randint(L)
            spot = spins_list[row][col]

            if spot == 1:
                dS = -2
            else:
                dS = 2

            dE = 2 * spot * (spins_list[(row + 1) % L][col] + spins_list[(row - 1) % L][col]  + spins_list[row][(col + 1) % L] + spins_list[row][(col - 1) % L])

            if dE < 0:
                spins_list[row][col] += dS 
                E += dE
                S += dS
            else:
                if np.random.rand(1) < np.exp(-dE / temp):
                    spins_list[row][col] += dS
                    E += dE
                    S += dS
                    
        iplus1 = S_avg[i] + 1/(i+1) * (S - S_avg[i])
        S_avg.append(iplus1) 
    
    avg = sum(S_avg) / len(S_avg)
    
    M = avg * L ** -2
        
    return spins_list, E, S, M


def two_dim_ising_grid(L, temp, num_steps):
    '''
    Two dimensional structure of spins. A MCMC simulation is run to test how changes in temperature will result in spin
    conglomeration.
    
    Arguments:
    L (int): Lattice of side length (number of length)
    temp (int): given temperature
    num_steps (int): number of updates to make in the MCMC update.
    
    Returns:
    spins_list (int numpy ndarray): array of spins values (1 or -1) replicating a lattice structure
    '''
    
    S = 0
    E = 0
    
    spins_list = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            spins_list[i][j] = np.random.choice([-1, 1])
            S += spins_list[i][j]
            
    for i in range(L):
        for j in range(L): # -= because of the negative sum
            E -= spins_list[i][j] * (spins_list[(i + 1) % L][j] + spins_list[(i - 1) % L][j]  + spins_list[i][(j + 1) % L] + spins_list[i][(j - 1) % L])
        
    for i in range(num_steps):
        row = np.random.randint(L)
        col = np.random.randint(L)
        spot = spins_list[row][col]

        if spot == 1:
            dS = -2
        else:
            dS = 2

        dE = 2 * spot * (spins_list[(row + 1) % L][col] + spins_list[(row - 1) % L][col]  + spins_list[row][(col + 1) % L] + spins_list[row][(col - 1) % L])

        if dE < 0:
            spins_list[row][col] += dS 
            E += dE
            S += dS
        else:
            if np.random.rand(1) < np.exp(-dE / temp):
                spins_list[row][col] += dS
                E += dE
                S += dS
    
    return spins_list
