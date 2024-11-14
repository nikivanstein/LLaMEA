import numpy as np

class OptimizedFusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget, self.dim, self.num_particles, self.alpha, self.beta, self.gamma = budget, dim, num_particles, alpha, beta, gamma

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_process():
            nonlocal best_global_pos
            r1, r2 = np.random.rand(2)
            velocity[:] = self.alpha * velocity + self.beta * r1 * (best_global_pos - population) + self.gamma * r2 * (population - np.mean(population, axis=0))
            population[:] += velocity

        population, velocity = initialize_particles(), np.zeros((self.num_particles, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_global_pos = population[np.argmin(fitness)]

        for _ in range(self.budget - self.num_particles):
            update_process()
            fitness = np.array([func(individual) for individual in population])
            best_global_pos = population[np.argmin(fitness)]

        return best_global_pos