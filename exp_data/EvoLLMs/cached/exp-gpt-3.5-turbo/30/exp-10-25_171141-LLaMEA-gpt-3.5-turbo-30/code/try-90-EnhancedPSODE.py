# import numpy as np

class EnhancedPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.particles = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.pbest = self.particles.copy()
        self.pbest_scores = np.full(self.pop_size, np.inf)
        self.gbest = np.zeros(self.dim)
        self.gbest_score = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                fitness = func(self.particles[i])
                if fitness < self.pbest_scores[i]:
                    self.pbest[i] = self.particles[i]
                    self.pbest_scores[i] = fitness
                    if fitness < self.gbest_score:
                        self.gbest = self.pbest[i]
                        self.gbest_score = fitness
                w = 0.5 + 0.2 * np.random.rand()
                c1 = 1.5 * np.random.rand()
                c2 = 1.5 * np.random.rand()
                self.velocities[i] = w * self.velocities[i] + c1 * np.random.rand() * (self.pbest[i] - self.particles[i]) + c2 * np.random.rand() * (self.gbest - self.particles[i])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], -5.0, 5.0)
                # Dynamic Adaptive Mutation Differential Evolution (AMDE)
                F = 0.5 + 0.5 * np.random.rand()
                CR = 0.1 + 0.9 * np.random.rand()
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = self.particles[r1] + F * (self.particles[r2] - self.particles[r3])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, self.particles[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness:
                    self.particles[i] = trial
        return self.gbest