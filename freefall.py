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

    def getNext(self, dragType=0):
        self.time += self.deltat
        
        # Calc new velocity
        dragforce = 0
        if (dragType == 1):
            dragforce = abs(self.drag_coeff * self.vel)
        elif (dragType == 2):
            dragforce = rho/2 * self.vel * self.vel * self.area * self.drag_coeff

        dvel = (-g + dragforce/self.mass) * self.deltat
        avvel = self.vel + dvel/2
        self.vel += dvel

        # Calculate position with average position during frame
        self.pos += avvel * self.deltat
        return (self.pos, self.vel, self.time)

    def predict(self, time, dragType=0):
        # Get number of frames to simulate
        n = math.ceil(time / self.deltat)
        frames = []
        for _ in range(n):
            frames += [self.getNext(dragType)]

        return frames

def plotdata(filename):
    data = np.loadtxt(filename, skiprows=1)

    t = data[:, 0]
    t -= t[0] - 0.03 # t offset
    y = data[:, 1]
    y -= y[0] + 0.002 # y offset

    plt.plot(t, y, 'o-')



def main():
    foam = Ball(m=0.00485, A=0.00477663773)
    out = foam.predict(.482, 2)

    plt.plot(np.array([el[2] for el in out]), np.array([el[0] for el in out]))
    # plotdata('data/foam/Foam_1.txt')
    # plotdata('data/foam/Foam_2.txt')
    # plotdata('data/foam/Foam_3.txt')
    plotdata('data/foam/Foam_4.txt')
    # plotdata('data/foam/Foam_5.txt')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.show()

if (__name__ == "__main__"):
    main()