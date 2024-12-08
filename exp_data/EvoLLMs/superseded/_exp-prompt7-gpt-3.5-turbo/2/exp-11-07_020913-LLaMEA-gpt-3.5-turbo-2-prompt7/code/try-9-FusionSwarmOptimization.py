import numpy as np

class FusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        self.velocity = np.zeros((self.num_particles, self.dim))

    def __call__(self, func):
        fitness = np.array([func(individual) for individual in self.population])
        best_global_pos = self.population[np.argmin(fitness)]
        for _ in range(self.budget - self.num_particles):
            r1, r2 = np.random.rand(2)
            self.velocity = self.alpha * self.velocity + self.beta * r1 * (best_global_pos - self.population) + self.gamma * r2 * (self.population - self.population.mean(axis=0))
            self.population += self.velocity
            fitness = np.array([func(individual) for individual in self.population])
            best_global_pos = self.population[np.argmin(fitness)]
        return best_global_pos