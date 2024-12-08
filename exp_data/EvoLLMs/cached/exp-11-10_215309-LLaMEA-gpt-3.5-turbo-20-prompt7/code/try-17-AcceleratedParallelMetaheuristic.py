import numpy as np
from concurrent.futures import ThreadPoolExecutor

class AcceleratedParallelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1

    def evaluate_solution(self, func, solution):
        return func(solution)

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(self.budget):
                mutation_strength = np.random.uniform(0, 1)
                candidate_solution = best_solution + mutation_strength * np.random.standard_normal(self.dim)
                futures.append(executor.submit(self.evaluate_solution, func, candidate_solution))
            
            fitness_values = np.array([future.result() for future in futures])
            fitness_variance = np.var(fitness_values)
            
            for idx, candidate_fitness in enumerate(fitness_values):
                if candidate_fitness < best_fitness:
                    best_solution = futures[idx].result()
                    best_fitness = candidate_fitness
                
                mutation_strength = np.clip(0.9 * mutation_strength + 0.1 * fitness_variance, 0, 1)
                candidate_solution = best_solution + mutation_strength * np.random.standard_normal(self.dim)
        
        return best_solution