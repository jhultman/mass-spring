import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(23)


"""
Notation:
    m:  Masses (kg).
    k:  Spring constants (N/m).
    x0: Initial positions (m).
    v0: Initial velocities (m/s).
    l:  Natural spring elongations (m).
    dt: Time discretization (s).
"""


class MassSpringSystem:

    def __init__(self):
        dt = 0.2
        n_iters = 200
        v0 = np.zeros(4)
        x0 = np.random.rand(4).cumsum()
        m = np.array([2, 1, 3, 2], np.float32)
        k = np.array([1.2, 1.4, 1.5], np.float32)
        l = np.random.uniform(0.7, 1.3, size=3) * np.diff(x0)
        self.params = (x0, v0, m, k, l, dt, n_iters)

    @staticmethod
    def forces(x, k, l):
        f_ij = k * (np.diff(x) - l)
        A = np.array([
            [+1,  0,  0],
            [-1, +1,  0],
            [ 0, -1, +1],
            [ 0,  0, -1],
        ])
        f = A @ f_ij
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
        self.ax.set(xlim=[xmin - border, xmax + border], ylim=[-1, 1])

    def init_ani(self):
        self.init_lims()
        self.scatter = self.ax.scatter([], [], animated=True, zorder=2)
        self.scatter.set_array(np.zeros(4))
        plot = lambda c: self.ax.plot([], [], linestyle='dashed', c=c, zorder=1)
        self.lines = [plot(c)[0] for c in 'rgb']
        return (self.scatter, *self.lines)

    def step_ani(self, frame):
        inds = np.s_[4 * frame : 4 * (frame + 1)]
        data = np.array([self.x[inds], self.y[inds]])
        [self.lines[i].set_data(data[:, [i, i + 1]]) for i in range(3)]
        self.scatter.set_offsets(data.T)
        return (self.scatter, *self.lines)

    def ani(self):
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
