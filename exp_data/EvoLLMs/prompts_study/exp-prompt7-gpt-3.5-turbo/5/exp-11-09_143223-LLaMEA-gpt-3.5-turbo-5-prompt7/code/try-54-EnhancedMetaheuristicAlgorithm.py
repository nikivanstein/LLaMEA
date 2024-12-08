import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size
        self.population_size = 10

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            population = [best_solution + self.mutation_step * np.random.uniform(-1, 1, self.dim) for _ in range(self.population_size)]
            population = [np.clip(candidate_solution, -5.0, 5.0) for candidate_solution in population]
            population_fitness = [func(candidate_solution) for candidate_solution in population]

            best_candidate_idx = np.argmin(population_fitness)
            if population_fitness[best_candidate_idx] < best_fitness:
                best_solution = population[best_candidate_idx]
                best_fitness = population_fitness[best_candidate_idx]
            
            # Adaptive mutation step size adjustment
            if np.random.rand() < 0.1:  # Probability of step size adjustment
                self.mutation_step *= np.exp(0.1 * np.random.uniform(-1, 1))
                self.mutation_step = max(0.1, min(self.mutation_step, 2.0))

        return best_solution