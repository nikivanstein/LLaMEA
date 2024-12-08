import numpy as np

class ParticleSwarmEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.pbest = np.zeros((self.population_size, self.dim))
        self.rbest = np.zeros((self.population_size, self.dim))
        self.pbest_fitness = np.zeros(self.population_size)
        self.rbest_fitness = np.zeros(self.population_size)
        self.ps = np.zeros((self.population_size, self.dim))
        self.rs = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Selection
            self.pbest_fitness = np.min(self.pbest[:, 0])
            self.rbest_fitness = np.min(self.rbest[:, 0])
            self.ps = self.pbest[np.argmin(self.pbest_fitness), :]
            self.rs = self.rbest[np.argmin(self.rbest_fitness), :]

            # Particle Swarm Optimization
            for i in range(self.population_size):
                r = np.random.uniform(0, 1, size=self.dim)
                self.ps[i] = self.ps[i] + r * (self.rbest[i, :] - self.ps[i])
                self.rs[i] = self.rs[i] + r * (self.best_candidate - self.rs[i])
                self.pbest[i, :] = self.ps[i]
                self.pbest_fitness[i] = func(self.ps[i])

                # Evolutionary Strategy
                self.candidates[i, :] = self.ps[i] + np.random.uniform(-0.25, 0.25, size=self.dim)

                # Check if the best candidate is improved
                if self.pbest_fitness[i] < self.best_fitness:
                    self.best_candidate = self.ps[i]
                    self.best_fitness = self.pbest_fitness[i]

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

particle_ESO = ParticleSwarmEvolutionaryStrategy(budget=100, dim=2)
best_candidate, best_fitness = particle_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")