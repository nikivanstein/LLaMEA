import numpy as np

class SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.particles = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.pbest = self.particles.copy()
        self.gbest = np.zeros(self.dim)
        self.gbest_score = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                fitness = func(self.particles[i])
                if fitness < func(self.pbest[i]):
                    self.pbest[i] = self.particles[i]
                    if fitness < self.gbest_score:
                        self.gbest = self.pbest[i]
                        self.gbest_score = fitness
                scale_factor = np.random.uniform(0.5, 2.0)
                crossover_rate = np.random.uniform(0.1, 0.9)
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = self.particles[r1] + scale_factor * (self.particles[r2] - self.particles[r3])
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.particles[i])
                if func(trial) < fitness:
                    self.particles[i] = trial
        return self.gbest