import numpy as np

class StreamlinedFusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget, self.dim, self.num_particles, self.alpha, self.beta, self.gamma = budget, dim, num_particles, alpha, beta, gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_global_pos = population[np.argmin(fitness)]

        for _ in range(self.budget - self.num_particles):
            r1, r2 = np.random.rand(2, self.num_particles, self.dim)
            velocity = self.alpha * velocity + self.beta * r1 * (best_global_pos - population) + self.gamma * r2 * (population - np.mean(population, axis=0))
            new_population = population + velocity
            new_fitness = np.apply_along_axis(func, 1, new_population)
            idx = np.argmin(new_fitness)
            if new_fitness[idx] < fitness.min():
                best_global_pos = new_population[idx]
            population, fitness = new_population, new_fitness

        return best_global_pos