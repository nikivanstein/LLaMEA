import numpy as np

class StochasticBinaryInspiredSearchAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.mutation_prob = mutation_prob

    def binary_mutation(self, population):
        mutation_mask = np.random.choice([0, 1], size=population.shape, p=[1-self.mutation_prob, self.mutation_prob])
        mutated_population = population + mutation_mask * np.random.uniform(-0.5, 0.5, size=population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.binary_mutation(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            population = population[top_indices]
            best_individual = population[np.argmin(fitness_values)]
            population = 0.8 * best_individual + 0.2 * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution