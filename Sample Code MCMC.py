# Please note that lines including $ should be formatted in LaTeX

import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt

## Simulate a Biased Coin
####Here is some LaTeX formatting, if you really want a better explanation of how the biased coin works, format this.
We can use induction. Let's start with $k = 1$:

$$\langle O \rangle_2 = \langle O \rangle_1 + \frac{1}{2} (o_2 - \langle O \rangle_1) \Rightarrow o_1 + \frac{1}{2}(o_2 - o_1) = \frac{1}{2}(o_1 + o_2)$$

This is a good sum. We can then assume if it works once, we can assume it will work again.

We will use this in the future:
$$ (1) \; \; \langle O \rangle_{k-1} = \left( \frac{1}{k - 1} \right) (k \langle O \rangle_k - o_k)$$.

Then for $k + 1$:

$$\langle O \rangle_{k+1} = \langle O \rangle_{k+1} + \left(  \frac{1}{k+1} \right) (o_{k + 1} - \langle O \rangle_k)$$

$$\langle O \rangle_{k+1} = \langle O \rangle_{k-1} + \frac{1}{k}(o_n - \langle O \rangle_{k-1}) + \left( \frac{1}{k+1}\right) \left( o_{k+1} - \langle O \rangle_{k-1} - \frac{o_k - \langle O \rangle_{k-1}}{k} \right)$$

$$\langle O \rangle_{k+1} = \langle O \rangle_{k-1} + \frac{o_k - \langle O \rangle_{k-1}}{k} + \left( \frac{o_{k+1} - \langle O \rangle_{k-1}}{k + 1} \right) $$

$$\langle O \rangle_{k + 1} = \langle O \rangle_{k-1} + \left( \frac{o_k - \langle O \rangle_{k-1} + o_{k+1} - \langle O \rangle_{k-1}}{k + 1} \right)$$

$$\langle O \rangle_{k+1} = \langle O \rangle_{k-1} - \left(  \frac{2}{k+1}\right) \langle O \rangle_{k-1} + \left( \frac{o_k + o_{k+1}}{k+1} \right)$$

$$\langle O \rangle_{k+1} = \left( \frac{k - 1}{k+1} \right) \langle O \rangle_{k+1} + \left( \frac{o_k + o_{k+1}}{k+1} \right)$$

Apply (1) here.

$$\langle O \rangle_{k+1} = \left( \frac{k\langle O \rangle_k - o_k}{k+1} \right) + \left( \frac{o_k + o_{k+1}}{k+1} \right) $$

$$ (2) \; \; \langle O \rangle_{k+1} = \frac{k \dot \langle O \rangle_k + o_{k+1}}{k+1} $$

Since an average is the sum of all numbers divided by the count of the numbers, $\langle O \rangle_k = \frac{o_1 + o_2 + o_3 + ... + o_k}{k} \Rightarrow k\langle O \rangle_k = o_1 + o_2 + o_3 + ... + o_k$,

We can apply (2) here to then find that
$\langle O \rangle_{k+1} = \frac{o_1 + o_2 + o_3 + ... + o_{k+1}}{k+1}$



print(weighted_coin(0.1, 100000, 2000))


## Simulated a weighted die
At about 75000 iterations, the value approaches roughly $(0.195 +/- 0.01).
Expected value is $20.00.

print(weighted_die(75000))
print(average_earnings_per_flip(.6))



## Plotting time series of intensive quantities
For the 16 length system, the magnetization seems to really converge around 100 steps, with the U converging in 50.
For the 32 length system, the magnetization seems to converge around 80 with the U converging around 60.

Perhaps it might be my code, but there does seem to be a slight difference in convergence depending on size, but nothing substantial

t, U, M = two_dim_ising(16, 3, 1000)
t2, U2, M2 = two_dim_ising(32, 3, 1000)fig = plt.figure(figsize = (13, 13))
fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
    
ax1.plot(t, U, label = "U")
ax1.plot(t, M, label = "M")
ax1.set_xlabel("Steps")
ax1.set_ylabel("U and M")
ax1.set_title("Internal Energy (U) and Magnetization (M) in 16x16 Lattice")
ax1.grid()
ax1.legend()

ax2.plot(t2, U2, label = "U")
ax2.plot(t2, M2, label = "M")
ax2.set_xlabel("Steps")
ax2.set_ylabel("U and M")
ax2.set_title("Internal Energy (U) and Magnetization (M) in 32x32 Lattice")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()



## Magnetization curves for different lattice sizes
temps = np.linspace(5, 0.001, 100)
S = 0
E = 0
Ls = [8, 16, 34, 64]
Ms = [[] for i in range(len(Ls))]

k = 0
for L in Ls:
    spins_list = np.zeros((L,L))

    for i in range(L):
        for j in range(L):
            spins_list[i][j] = np.random.choice([-1, 1])
            S += spins_list[i][j]

    for i in range(L):
        for j in range(L): # -= because of the negative sum
            E -= spins_list[i][j] * (spins_list[(i + 1) % L][j] + spins_list[(i - 1) % L][j]  + spins_list[i][(j + 1) % L] + spins_list[i][(j - 1) % L])
    
    for i in temps:
        if i > 2.2692:
            spins_list, E, S, M = two_dim_ising_temp(spins_list, E, S, L, i, 1000)
        else:
            t, pot, mag = two_dim_ising(L, i, 1000)
            M = sum(mag) / len(mag) 
            

        Ms[k].append(np.abs(M))
    
    plt.plot(temps, Ms[k], label = f"{L}")
    k += 1
    
with np.errstate(over='ignore'):
    plt.plot(temps, M_theoretical(temps), label = "Theoretical")
plt.legend()
plt.grid()
plt.show()





## Typical spin configuration at different temperatures
Below critical temperature, the spin configurations seems to climp together in groups of like spin. It is primarily in one direction, making it ferromagnetic.
At around critical temperature, the clumps seem to be thinner. 
Past critical temperature, the clumps are very small, if even there at all. Past Tc, the spins seems to be very disbursed.


size = 64
temps = [1.8, 2.3, 4.0]

for t in temps:
    spin = two_dim_ising_grid(size, t, 1000000)
    plt.imshow(spin, cmap = 'gray')
    plt.show()
