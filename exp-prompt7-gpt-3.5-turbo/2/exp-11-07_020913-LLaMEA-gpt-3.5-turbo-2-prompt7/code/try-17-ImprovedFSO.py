import numpy as np

class ImprovedFSO:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget, self.dim, self.num_particles, self.alpha, self.beta, self.gamma = budget, dim, num_particles, alpha, beta, gamma

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))

        def evaluate_fitness(population):
            return np.array(list(map(func, population)))

        def update_position_velocity(best_global_pos, population, velocity):
            r = np.random.rand(2)
            for i in range(self.num_particles):
                velocity[i] = self.alpha * velocity[i] + self.beta * r[0] * (best_global_pos - population[i]) + self.gamma * r[1] * (population[i] - np.mean(population, axis=0))
                population[i] += velocity[i]

        population, velocity = initialize_particles(), np.zeros((self.num_particles, self.dim))
        fitness = evaluate_fitness(population)
        best_global_pos = population[np.argmin(fitness)]

        for _ in range(self.budget - self.num_particles):
            update_position_velocity(best_global_pos, population, velocity)
            fitness = evaluate_fitness(population)
            best_global_pos = population[np.argmin(fitness)]

        return best_global_pos