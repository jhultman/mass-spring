import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(23)


class MassSpringSystem:

    def __init__(self):
        # Initial displacements of four particles placed along a line (m).
        x0 = np.random.rand(4).cumsum()

        # Initial velocities (m/s).
        v0 = np.zeros(4)

        # Masses of the four particles (kg).
        m = np.array([2, 1, 3, 2], np.float32)

        # Constants of springs connecting adjacent particles (N/m).
        k = np.array([1.2, 1.4, 1.5], np.float32)

        # Natural elongations of the three springs (m).
        delta = 0.3
        scale = np.random.uniform(1 - delta, 1 + delta, size=3)
        l = scale * np.diff(x0)

        # Time discretization (s).
        dt = 0.2

        # Num iterations.
        n_iters = 200

        self.params = (x0, v0, m, k, l, dt, n_iters)

    @staticmethod
    def forces(x, k, l):
        f_ij = k * (np.diff(x) - l)
        f0 = f_ij[0]
        f1 = f_ij[1] - f_ij[0]
        f2 = f_ij[2] - f_ij[1]
        f3 = -f_ij[2]
        f = np.hstack((f0, f1, f2, f3))
        return f

    def simulate(self):
        (x0, v0, m, k, l, dt, n_iters) = self.params 
        x, v = x0, v0
        x_history = []

        for _ in range(n_iters):
            f = MassSpringSystem.forces(x, k, l)
            a = f / m
            v = v + 0.5 * dt * a
            x = x + dt * v
            x_history += [x]

        x_history = np.array(x_history).flatten()
        y = np.zeros_like(x_history)
        return x_history, y


class Animator:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(4, 1))
        self.fig.tight_layout()

    def init_lims(self):
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        border = 0.1 * (xmax - xmin)
        self.ax.set_xlim(xmin - border, xmax + border)
        self.ax.set_ylim(-1, 1)

    def init_ani(self):
        self.init_lims()
        color = np.linspace(0, 255, 4)
        self.scatter = self.ax.scatter([], [], animated=True)
        self.scatter.set_array(color)
        return self.scatter,

    def step_ani(self, frame):
        inds = np.s_[4 * frame : 4 * (frame + 1)]
        data = np.array([self.x[inds], self.y[inds]])
        self.scatter.set_offsets(data.T)
        return self.scatter,

    def ani(self):
        args = ()
        animation = FuncAnimation(
            self.fig, 
            self.step_ani, 
            frames=range(len(self.x) // 4),
            init_func=self.init_ani, 
            blit=True, 
            interval=50,
            repeat=False,
        )
        animation.save('../images/springs.gif', dpi=80, writer='imagemagick')


def main():
    mass_spring = MassSpringSystem()
    x, y = mass_spring.simulate()
    animator = Animator(x, y)
    animator.ani()


if __name__ == '__main__':
    main()
