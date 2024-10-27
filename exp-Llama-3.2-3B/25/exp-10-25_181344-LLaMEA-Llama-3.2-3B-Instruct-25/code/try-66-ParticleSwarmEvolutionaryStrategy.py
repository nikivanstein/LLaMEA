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
        self.ps = np.zeros((self.population_size, self.dim))
        self.rs = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Particle Swarm Optimization
            for i in range(self.population_size):
                self.pbest[i] = self.candidates[i, :]
                self.rbest[i] = self.candidates[i, :]
                self.ps[i] = self.rbest[i] + np.random.uniform(-0.1, 0.1, size=self.dim)
                self.rs[i] = self.pbest[i] + np.random.uniform(-0.1, 0.1, size=self.dim)
                new_fitness = func(self.ps[i]) + np.random.uniform(-0.1, 0.1, size=self.dim)
                if new_fitness < self.rbest[i, 0]:
                    self.rbest[i, :] = self.ps[i, :]
                    self.rs[i, :] = self.rbest[i, :]
                    self.candidates[i, :] = self.rs[i, :]
                    if new_fitness < self.best_fitness:
                        self.best_candidate = self.candidates[i, :]
                        self.best_fitness = new_fitness

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

ps_es = ParticleSwarmEvolutionaryStrategy(budget=100, dim=2)
best_candidate, best_fitness = ps_es(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")