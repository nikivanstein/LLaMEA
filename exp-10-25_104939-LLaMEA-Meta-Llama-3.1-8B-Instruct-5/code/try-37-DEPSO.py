import numpy as np
import random

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.max_iter = 100
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.particles = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best_particles = np.copy(self.particles)
        self.best_fitness = np.array([float('inf')] * self.pop_size)

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.pop_size):
                x = np.copy(self.particles[j])
                for k in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    if r1 < 0.05:
                        # Refine the strategy of the selected solution
                        idx = np.random.choice(self.pop_size, 1, replace=False, p=self.best_fitness / np.sum(self.best_fitness))
                        x[k] = self.best_particles[idx][0][k] + self.w * (self.best_particles[idx][0][k] - self.particles[np.random.randint(0, self.pop_size), k])
                    elif r2 < 0.5:
                        x[k] = self.particles[j, k] + self.w * (self.particles[j, k] - self.particles[np.random.randint(0, self.pop_size), k])
                    else:
                        x[k] = self.best_particles[j, k] + self.w * (self.best_particles[j, k] - self.particles[np.random.randint(0, self.pop_size), k])
                fitness = func(x)
                if fitness < self.best_fitness[j]:
                    self.best_fitness[j] = fitness
                    self.best_particles[j] = x
            if np.mean(self.best_fitness) < np.mean(self.best_fitness):
                self.particles = np.copy(self.best_particles)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {np.mean(self.best_fitness)}")
        return self.best_particles[np.argmin(self.best_fitness)]