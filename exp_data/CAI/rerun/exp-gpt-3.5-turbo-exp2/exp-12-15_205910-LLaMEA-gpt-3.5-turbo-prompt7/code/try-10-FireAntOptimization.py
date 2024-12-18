import numpy as np

class FireAntOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_solution = best_solution + np.random.uniform(-1, 1, self.dim)
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
            
            # Opposition-based learning
            opposite_solution = self.lower_bound + self.upper_bound - best_solution
            opposite_fitness = func(opposite_solution)
            if opposite_fitness < best_fitness:
                best_solution = opposite_solution
                best_fitness = opposite_fitness
        
        return best_solution