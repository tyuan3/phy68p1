import math
from freefall import Ball

g = 9.8 # m/s^2

foam = Ball(m=0.00485, A=0.00477663773)

def linear_drag(m, k, t): 
    m = foam.mass # kg
    k = foam.lin_drag_coeff # linear drag coefficient
    v_t = (m * g) / k # terminal velocity

    return v_t * t - ((v_t * m)/ k) * (1 - math.exp(-(k*t)/m))

def quadratic_drag(m, c, t):
    m = foam.mass
    c = foam.quad_drag_coeff
    v_t = math.sqrt((m * g) / c)
    
    return ((v_t ** 2) / g) * math.log(math.cosh((g / v_t) * t))

import matplotlib.pyplot as plt
import numpy as np

t_vals = np.linspace(0, 0.5, 200)  # zoom into 0 to 0.1s instead of 0 to 2s

# Evaluate each function at every t (can't just call with array since math.exp/log need scalars)
linear = [-linear_drag(foam.mass, foam.lin_drag_coeff, t) for t in t_vals]
quadratic = [-quadratic_drag(foam.mass, foam.quad_drag_coeff, t) for t in t_vals]

plt.plot(t_vals, linear, label="Linear drag")
plt.plot(t_vals, quadratic, label="Quadratic drag")
plt.xlabel("Time (s)")
plt.ylabel("Distance fallen (m)")
plt.title("Freefall distance vs. time")
plt.legend()
plt.grid(True)
plt.show()
