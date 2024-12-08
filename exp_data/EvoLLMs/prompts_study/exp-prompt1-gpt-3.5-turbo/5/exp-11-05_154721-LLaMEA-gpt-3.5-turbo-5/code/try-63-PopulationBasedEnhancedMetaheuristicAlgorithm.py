import numpy as np

class PopulationBasedEnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim, population_size=10):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_scale = 0.5  # Initialize mutation scale

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]  # Initialize population
        
        for eval_count in range(self.budget):
            if eval_count % (self.budget // 10) == 0 and eval_count > 0:
                self.mutation_scale = 0.5 - 0.5 * eval_count / self.budget  # Adapt mutation scale

            candidate_solutions = [indiv + np.random.uniform(-self.mutation_scale, self.mutation_scale, self.dim) for indiv in population]  # Mutation step
            candidate_solutions = [np.clip(indiv, -5.0, 5.0) for indiv in candidate_solutions]  # Ensure solutions are within bounds
            candidate_fitnesses = [func(indiv) for indiv in candidate_solutions]
            
            best_candidate_idx = np.argmin(candidate_fitnesses)
            if candidate_fitnesses[best_candidate_idx] < best_fitness:
                best_solution = candidate_solutions[best_candidate_idx]
                best_fitness = candidate_fitnesses[best_candidate_idx]
            
            population = [population[i] + 0.6 * (candidate_solutions[i] - population[i]) for i in range(self.population_size)]  # Update population using best candidate
        
        return best_solution