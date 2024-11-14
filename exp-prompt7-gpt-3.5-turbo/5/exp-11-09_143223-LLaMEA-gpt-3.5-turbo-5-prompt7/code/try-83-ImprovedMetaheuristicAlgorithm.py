import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size
        self.mutation_prob = 0.5  # Initial mutation probability
        self.population_size = 10  # Initial population size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(0, self.budget, self.population_size):  # Adjusted for dynamic population size
            population_solutions = [best_solution + self.mutation_step * np.random.uniform(-1, 1, self.dim) for _ in range(self.population_size)]
            population_solutions = [np.clip(sol, -5.0, 5.0) for sol in population_solutions]
            population_fitness = [func(sol) for sol in population_solutions]
            
            best_index = np.argmin(population_fitness)
            if population_fitness[best_index] < best_fitness:
                best_solution = population_solutions[best_index]
                best_fitness = population_fitness[best_index]

            self.mutation_prob = max(0.1, min(self.mutation_prob + 0.05 * np.random.uniform(-1, 1), 0.9))
            self.mutation_step *= np.exp(0.1 * np.random.uniform(-1, 1))
            self.mutation_step = max(0.1, min(self.mutation_step, 2.0))
            self.population_size = max(5, min(self.population_size + 1, 20))  # Dynamic population size adjustment

        return best_solution