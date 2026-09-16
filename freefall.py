import math

import matplotlib.pyplot as plt
import numpy as np

g = 9.8

class Ball:
    # Takes mass (kg)
    def __init__(self, m, pos = 0, dt = 1/60) -> None:
        self.mass = m
        self.deltat = dt
        self.pos = pos
        self.vel = 0
        self.time = 0
        pass

    def getNext(self):
        self.time += self.deltat
        
        # Calc new velocity
        dvel = -g * self.deltat
        avvel = self.vel + dvel/2
        self.vel += dvel

        # Calculate position with average position during frame
        self.pos += avvel * self.deltat
        return (self.pos, self.vel, self.time)

    def predict(self, time):
        # Get number of frames to simulate
        n = math.ceil(time / self.deltat)
        frames = []
        for _ in range(n):
            frames += [self.getNext()]

        return frames

def plotdata(filename):
    data = np.loadtxt(filename, skiprows=1)

    t = data[:, 0]
    t -= t[0] # Makes sure data starts at t=0
    y = data[:, 1]
    y -= y[0] # Makes sure data starts at y=0

    plt.plot(t, y, 'o-')



def main():
    test = Ball(1)
    out = test.predict(.482)

    plt.plot(np.array([el[2] for el in out]), np.array([el[0] for el in out]))
    plotdata('data/foam/Foam_1.txt')
    plotdata('data/foam/Foam_2.txt')
    plotdata('data/foam/Foam_3.txt')
    plotdata('data/foam/Foam_4.txt')
    plotdata('data/foam/Foam_5.txt')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.show()

if (__name__ == "__main__"):
    main()