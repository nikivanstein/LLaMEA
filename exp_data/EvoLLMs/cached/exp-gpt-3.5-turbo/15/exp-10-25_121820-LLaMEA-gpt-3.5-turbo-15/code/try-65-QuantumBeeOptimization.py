import numpy as np

class QuantumBeeOptimization:
    def __init__(self, budget, dim, population_size=30, quantum_factor=0.6, flip_probability=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.quantum_factor = quantum_factor
        self.flip_probability = flip_probability

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                candidate_solution = population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.flip_probability:
                        candidate_solution[j] = np.random.uniform(-5.0, 5.0)
                    elif np.random.rand() < self.quantum_factor:
                        candidate_solution[j] = best_solution[j]
                
                candidate_fitness = func(candidate_solution)
                if candidate_fitness < fitness[i]:
                    population[i] = candidate_solution
                    fitness[i] = candidate_fitness

                    if candidate_fitness < func(best_solution):
                        best_solution = candidate_solution

        return best_solution