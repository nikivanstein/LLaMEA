import numpy as np
import random

class SwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.x = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.f = np.zeros(self.population_size)
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                # Evaluate the function
                f = func(self.x[i])
                # Update the personal best
                if f < self.f[i]:
                    self.f[i] = f
                    self.pbest[i] = self.x[i]
                # Update the global best
                if f < self.gbest:
                    self.gbest = f
                    self.x[:, :] = self.pbest

            # Perform DE and PSO updates
            for i in range(self.population_size):
                # Randomly select a target vector
                j = random.randint(0, self.population_size - 1)
                target = self.pbest[j] + self.x[i] - self.x[j]
                # Calculate the step size
                step_size = np.random.uniform(0.5, 1.5)
                # Update the target vector
                self.x[i] = self.x[i] + step_size * (target - self.x[i])

            # Evaluate the function again
            f = func(self.x[i])
            # Update the personal best and global best
            if f < self.f[i]:
                self.f[i] = f
                self.pbest[i] = self.x[i]
                if f < self.gbest:
                    self.gbest = f

# Example usage
def func(x):
    return np.sum(x**2)

swarm_de = SwarmDE(budget=100, dim=10)
func(swarm_de)