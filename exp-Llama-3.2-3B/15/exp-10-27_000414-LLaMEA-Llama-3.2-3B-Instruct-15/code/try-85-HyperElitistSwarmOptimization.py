import numpy as np
import random

class HyperElitistSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elite_size = int(budget * 0.2)
        self.swarm_size = budget - elite_size
        self.elite = np.zeros((self.elite_size, self.dim))
        self.swarm = np.zeros((self.swarm_size, self.dim))
        self.best = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize swarm and elite
            self.swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            self.elite = np.random.uniform(-5.0, 5.0, (self.elite_size, self.dim))

            # Evaluate function for each individual
            for i in range(self.swarm_size + self.elite_size):
                x = np.zeros(self.dim)
                if i < self.swarm_size:
                    x = self.swarm[i]
                else:
                    x = self.elite[i - self.swarm_size]
                f = func(x)
                if f < self.best:
                    self.best = f
                    self.elite[i - self.swarm_size] = x
                if f < func(self.elite[i % self.elite_size]):
                    self.elite[i % self.elite_size] = x

            # Perform selection
            selected_elite = []
            for i in range(self.elite_size):
                r = np.random.rand()
                if r < 0.5:
                    selected_elite.append(self.elite[i])
                else:
                    selected_elite.append(self.swarm[np.random.randint(0, self.swarm_size)])

            # Perform crossover
            new_elite = []
            for i in range(self.elite_size):
                if np.random.rand() < 0.15:
                    new_elite.append(selected_elite[np.random.randint(0, self.elite_size)])
                else:
                    new_elite.append(selected_elite[np.random.randint(0, self.elite_size)])

            # Perform mutation
            for i in range(self.elite_size):
                if np.random.rand() < 0.1:
                    new_elite[i] += np.random.uniform(-0.1, 0.1, self.dim)

            # Update elite and swarm
            self.elite = new_elite
            self.swarm = [x for x in self.elite if x not in selected_elite]

            # Evaluate function for each individual
            for i in range(self.swarm_size + self.elite_size):
                x = np.zeros(self.dim)
                if i < self.swarm_size:
                    x = self.swarm[i]
                else:
                    x = self.elite[i - self.swarm_size]
                f = func(x)
                if f < self.best:
                    self.best = f
                    self.elite[i - self.swarm_size] = x

        return self.best