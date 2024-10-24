import numpy as np
import random
import operator

class HybridPSAS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.particles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))
        self.pbest = self.particles.copy()
        self.swarm_temp = 1.0
        self.swarm_alpha = 0.7298
        self.swarm_beta = 0.1682
        self.swarm_gamma = 0.139
        self.swarm_delta = 0.011

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Evaluate the current particles
            f_particles = func(self.particles)

            # Update the best solution
            f_evals = f_particles[0]
            x_best = self.particles[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Update the pbest
            for i in range(len(self.particles)):
                if f_evals < self.pbest[i, 0]:
                    self.pbest[i, 0] = f_evals
                    self.pbest[i, 1] = self.particles[i]

            # Update the bounds
            self.bounds = np.array([np.min(self.particles, axis=0), np.max(self.particles, axis=0)])

            # Simulated annealing
            if self.swarm_temp > 1e-6:
                prob = np.random.rand()
                if prob < np.exp((self.swarm_temp - self.swarm_delta) * (f_evals - self.f_best)):
                    self.particles = self.pbest.copy()
                    self.swarm_temp *= self.swarm_alpha
                    self.f_evals = f_evals

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybridPSAS = HybridPSAS(budget=10, dim=2)
x_opt = hybridPSAS(func)
print(x_opt)