import numpy as np
from concurrent.futures import ThreadPoolExecutor

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size

    def evaluate_candidate(self, func, solution):
        return func(solution)

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                candidate_solutions = [best_solution + self.mutation_step * np.random.uniform(-1, 1, self.dim) for _ in range(4)]
                candidate_solutions = [np.clip(sol, -5.0, 5.0) for sol in candidate_solutions]
                
                candidate_fitnesses = list(executor.map(lambda x: self.evaluate_candidate(func, x), candidate_solutions))
                
                best_candidate_idx = np.argmin(candidate_fitnesses)
                if candidate_fitnesses[best_candidate_idx] < best_fitness:
                    best_solution = candidate_solutions[best_candidate_idx]
                    best_fitness = candidate_fitnesses[best_candidate_idx]
                
                if np.random.rand() < 0.1:
                    self.mutation_step *= np.exp(0.1 * np.random.uniform(-1, 1))
                    self.mutation_step = max(0.1, min(self.mutation_step, 2.0))
        
        return best_solution