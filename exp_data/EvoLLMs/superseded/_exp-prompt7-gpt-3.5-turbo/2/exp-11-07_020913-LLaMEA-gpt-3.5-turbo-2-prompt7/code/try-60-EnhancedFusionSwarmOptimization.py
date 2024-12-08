import numpy as np

class EnhancedFusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rand_vals = np.random.rand(3, self.num_particles, self.dim)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_global_pos = population[np.argmin(fitness)]

        for _ in range(self.budget - self.num_particles):
            r1, r2 = self.rand_vals[:2]
            velocity = self.alpha * velocity + self.beta * r1 * (best_global_pos - population) + self.gamma * r2 * (population - np.mean(population, axis=0))
            population += velocity
            fitness = np.array([func(individual) for individual in population])
            best_global_pos = population[np.argmin(fitness)]

        return best_global_pos