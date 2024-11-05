import numpy as np

class EnhancedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def local_search(self, func, solution, search_radius=0.1):
        current_fitness = func(solution)
        for _ in range(3):
            potential_solution = solution + np.random.uniform(-search_radius, search_radius, size=self.dim)
            if func(potential_solution) < current_fitness:
                solution = potential_solution
        return solution
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation_rate = np.random.uniform(0.01, 0.1)
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = best_solution + mutation
            new_solution = self.local_search(func, new_solution)  # Incorporating local search
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]