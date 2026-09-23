import math

import matplotlib.pyplot as plt
import numpy as np

g = 9.8 # m/s^2
rho = 1.225 # kg/m^3

class Ball:
    # Takes mass (kg)
    def __init__(self, m, A, b = 0.47, pos = 0, dt = 1/60) -> None:
        self.mass = m
        self.deltat = dt
        self.pos = pos
        self.vel = 0
        self.time = 0
        self.drag_coeff = b
        self.area = A
        pass

    def getNext(self, dragType=0, method='euler_improved'):
        self.time += self.deltat
        
        # Calc new velocity
        dragforce = 0
        if (dragType == 1):
            dragforce = abs(self.drag_coeff * self.vel)
        elif (dragType == 2):
            dragforce = rho/2 * self.vel * self.vel * self.area * self.drag_coeff

        # Standard Euler method using left-endpoint approximation
        if method == 'euler':
            dvel = (-g + dragforce/self.mass) * self.deltat
            self.pos += self.vel * self.deltat # Compute next position w/ left endpoint
            self.vel += dvel
            
        # Improved Euler method using average velocity of left and right endpoints
        elif method == 'euler_improved':
            dvel = (-g + dragforce/self.mass) * self.deltat
            avvel = self.vel + dvel/2
            self.vel += dvel

            # Calculate position with average position during frame
            self.pos += avvel * self.deltat

        elif method == 'RK4':
            ...

        return (self.pos, self.vel, self.time)

    def predict(self, time, dragType=0, method='euler_improved'):
        # Get number of frames to simulate
        n = math.ceil(time / self.deltat)
        frames = []
        for _ in range(n):
            frames += [self.getNext(dragType, method)]

        return frames

def plotdata(filename, n):
    y_all = []
    t_ref = None

    for i in range(1, n + 1): 
        data = np.loadtxt(f'{filename}_{i}.txt', skiprows=1)

        t = data[:, 0]
        t -= t[0] - 0.03  # t offset
        y = data[:, 1]
        y = -(y - y[0] - 0.002)  # y offset and flip sign
        if t_ref is None:
            t_ref = t  # use first trial's time array as the reference grid

        y_all.append(y)

    y_all = np.array(y_all)          # shape (n, N)
    y_mean = y_all.mean(axis=0)      # average at each timestep
    # y_std = y_all.std(axis=0)        # trial-to-trial spread

    plt.plot(t_ref, y_mean, 'o-', label=f"Average over {n} trials")
    # plt.fill_between(t_ref, y_mean - y_std, y_mean + y_std, alpha=0.2)

import analytical

def main():
    balls = [Ball(m=0.00485, A=0.00477663773),
             Ball(m=0.075, A=0.002299788928),
             Ball(m=0.0372, A=0.04171260073)]
    
    foam = balls[0]
    rubber = balls[1]
    beach = balls[2]

    # Compute quadratic solution and plot
    out = beach.predict(.482, 2, method='euler_improved')
    plt.plot(np.array([el[2] for el in out]), np.array([el[0] for el in out]), label="Euler (improved), quadratic Drag")

    out2 = beach.predict(.482, 2, method='euler')
    plt.plot(np.array([el[2] for el in out2]), np.array([el[0] for el in out2]), label="Euler, quadratic Drag")

    # Compute linear solution and plot
    # out2 = foam.predict(.482, 1, method='euler_improved')
    # plt.plot(np.array([el[2] for el in out2]), np.array([el[0] for el in out2]), label="Euler (improved), linear Drag")

    # Plot analytical solution
    # plt.plot(analytical.t_vals, analytical.linear, label="Linear drag")
    # plt.plot(analytical.t_vals, analytical.quadratic, label="Quadratic drag")
    
    balltype = 'beach' # enter type of ball in all lowercase
    plotdata(f'data/{balltype}/{balltype.capitalize()}', 5)

    plt.title(f'{balltype.capitalize()} Freefall with Drag')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Figures/{balltype}_2')
    plt.show()

if (__name__ == "__main__"):
    main()