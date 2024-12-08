import numpy as np

class DynamicPopulationSizeAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10  # Initialize population size
        self.mutation_scale = 0.5  # Initialize mutation scale

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for eval_count in range(self.budget):
            if eval_count % (self.budget // 10) == 0 and eval_count > 0:
                self.mutation_scale = 0.5 - 0.5 * eval_count / self.budget  # Adapt mutation scale
                self.population_size = 10 + eval_count // (self.budget // 10)  # Adjust population size dynamically

            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]  # Create population
            
            for candidate_solution in population:
                candidate_solution += np.random.uniform(-self.mutation_scale, self.mutation_scale, self.dim)  # Mutation step
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
                candidate_fitness = func(candidate_solution)
                
                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
        
        return best_solution