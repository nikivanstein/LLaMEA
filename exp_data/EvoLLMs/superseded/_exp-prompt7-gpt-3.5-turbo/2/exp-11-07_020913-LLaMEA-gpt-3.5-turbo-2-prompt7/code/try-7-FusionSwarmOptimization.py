import numpy as np

class FusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        
        def evaluate_fitness(population):
            return np.array([func(individual) for individual in population])
        
        def update_position_velocity(best_global_pos, population, velocity):
            r1, r2 = np.random.rand(2)
            for i in range(self.num_particles):
                velocity[i] = self.alpha * velocity[i] + self.beta * r1 * (best_global_pos - population[i]) + self.gamma * r2 * (population[i] - population.mean(axis=0))
                population[i] += velocity[i]
        
        population = initialize_particles()
        velocity = np.zeros((self.num_particles, self.dim))
        fitness = evaluate_fitness(population)
        best_global_pos = population[np.argmin(fitness)]
        
        for _ in range(self.budget - self.num_particles):
            update_position_velocity(best_global_pos, population, velocity)
            fitness = evaluate_fitness(population)
            best_global_pos = population[np.argmin(fitness)]
        
        return best_global_pos