import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            best_particle = self.particles[np.argmin([func(p) for p in self.particles])]
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                mutation = np.random.uniform(-1, 1, self.dim)
                self.velocities[i] = 0.5 * self.velocities[i] + 1.5 * r1 * (best_particle - self.particles[i]) + 2.0 * r2 * (self.particles[np.random.randint(0, self.population_size)] - self.particles[i])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] + mutation, -5.0, 5.0)
        return self.particles[np.argmin([func(p) for p in self.particles])]