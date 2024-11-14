import numpy as np

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size
        self.population_size = 10  # Initial population size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(int(self.budget * 1.05)):  # Increase the number of iterations by 5%
            candidate_solutions = [best_solution + self.mutation_step * np.random.uniform(-1, 1, self.dim) for _ in range(self.population_size)]
            candidate_solutions = np.clip(candidate_solutions, -5.0, 5.0)
            candidate_fitnesses = [func(candidate) for candidate in candidate_solutions]
            
            best_candidate_idx = np.argmin(candidate_fitnesses)
            if candidate_fitnesses[best_candidate_idx] < best_fitness:
                best_solution = candidate_solutions[best_candidate_idx]
                best_fitness = candidate_fitnesses[best_candidate_idx]
            
            if np.random.rand() < 0.1:  # Probability of step size adjustment
                self.mutation_step *= np.exp(0.1 * np.random.uniform(-1, 1))
                self.mutation_step = max(0.1, min(self.mutation_step, 2.0))
            
            if np.random.rand() < 0.05:  # Probability of population size adjustment
                self.population_size = max(5, min(self.population_size + int(np.random.normal(0, 1)), 20))

        return best_solution