# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 04:07:05 2024

@author: arunb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:28:56 2024

@author: arunb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def aat(x):
    return 1-(x**2)
def bat(x):
    return -x


def sigma(y, t, x):
    return y*t+x
def gamma(x, t, y):
    return bat(x)*( np.exp(t *aat(x))-1)/(aat(x)) + y* np.exp(t *aat(x))
# Define initial values for x and y
x0 = -.2 # Initial value for x
y0 = 0 # Initial value for y

# Define time and number of iterations
t =25  # Total time
n = 25 # Number of iterations


# Initialize arrays to store the solutions
x = np.zeros(n+1)
y = np.zeros(n+1)


# Set initial values
x[0] = x0
y[0] = y0

# Time step
dt =t / n

#Sixth
def compute_a_k(k):
    # Compute the exponent for the complex number
    aaa=np.exp((1j * np.pi) / (2 * k + 1)) /(2**(1 / (2 * k + 1)) + 2 * np.exp((1j * np.pi )/ (2 * k + 1)))
    return aaa

aa=compute_a_k(1) * compute_a_k(2)
bb=compute_a_k(2)*(1-2*compute_a_k(1))
cc=compute_a_k(1)*(1-2*compute_a_k(2))
dd=(1-2*compute_a_k(1))*(1-2*compute_a_k(2))

for k in range(n):
    # Apply each step of the composition in reverse order
    # Apply e^{a \frac{t}{2} \K_2}
    y_k_1 = gamma(x[k], aa * dt/2, y[k])

    # Apply e^{a t \K_1}
    x_k_2 = sigma(y_k_1, aa * dt, x[k])

    # Apply e^{(a+b) \frac{t}{2} \K_2}
    y_k_3 = gamma(x_k_2, (aa + bb) * dt/2, y_k_1)

    # Apply e^{b t \K_1}
    x_k_4 = sigma(y_k_3, bb * dt, x_k_2)

    # Apply e^{(a+b) t \K_2}
    y_k_5 = gamma(x_k_4, (aa + bb) * dt/2, y_k_3)

    # Apply e^{a t \K_1}
    x_k_6 = sigma(y_k_5, aa * dt, x_k_4)

    # Apply e^{(a+c) \frac{t}{2} \K_2}
    y_k_7 = gamma(x_k_6, (aa + cc) * dt/2, y_k_5)

    # Apply e^{c t \K_1}
    x_k_8 = sigma(y_k_7, cc * dt, x_k_6)

    # Apply e^{(c+d) \frac{t}{2} \K_2}
    y_k_9 = gamma(x_k_8, (cc + dd) * dt/2, y_k_7)

    # Apply e^{d t \K_1}
    x_k_10 = sigma(y_k_9, dd * dt, x_k_8)

    # Apply e^{(c+d) \frac{t}{2} \K_2}
    y_k_11 = gamma(x_k_10, (cc + dd) * dt/2, y_k_9)

    # Apply e^{c t \K_1}
    x_k_12 = sigma(y_k_11, cc * dt, x_k_10)

    # Apply e^{(a+c) \frac{t}{2} \K_2}
    y_k_13 = gamma(x_k_12, (aa + cc) * dt/2, y_k_11)

    # Apply e^{a t \K_1}
    x_k_14 = sigma(y_k_13, aa * dt, x_k_12)

    # Apply e^{(a+b) t \K_2}
    y_k_15 = gamma(x_k_14, (aa + bb) * dt/2, y_k_13)

    # Apply e^{b t \K_1}
    x_k_16 = sigma(y_k_15, bb * dt, x_k_14)

    # Apply e^{(a+b) \frac{t}{2} \K_2}
    y_k_17 = gamma(x_k_16, (aa + bb) * dt/2, y_k_15)

    # Apply e^{a t \K_1}
    x[k+1] = sigma(y_k_17, aa * dt, x_k_16)

    # Apply e^{a \frac{t}{2} \K_2}
    y[k+1] = gamma( x[k+1], aa * dt/2, y_k_17)



# Differential equations
def system_of_eqs(t, z):
    x, y = z
    dxdt = y
    dydt = aat(x)*y + bat(x)
    return [dxdt, dydt]

rtol = 1e-2 # Relative tolerance RK45

# Solve the system using solve_ivp
t_span = (0, t)
t_eval = np.linspace(0, t, n+1)
sol = solve_ivp(system_of_eqs, t_span, [x0, y0], method="RK45", t_eval=t_eval, rtol=rtol)

# Extract the solutions
x_ode = sol.y[0]
y_ode = sol.y[1]






plt.figure(figsize=(10, 6))

# Phase plot for iterative method
plt.plot(x, y, 'r--', label='Method Phase Plot')

#Phase plot for ODE RK45 solution
plt.plot(x_ode, y_ode, 'b-', label='RK45 ODE Solution Phase Plot')

plt.title('Phase Plot: x vs. y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


