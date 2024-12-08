import numpy as np

class DifferentialGravitationalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        G = 6.67430e-11  # Gravitational constant
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            for _ in range(self.budget):
                solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                fitness = func(solution)
                
                if fitness < best_fitness:
                    best_solution = solution
                    best_fitness = fitness
                else:
                    # Differential evolution
                    mutant = best_solution + 0.5 * (best_solution - solution)
                    mutant_fitness = func(mutant)
                    if mutant_fitness < best_fitness:
                        best_solution = mutant
                        best_fitness = mutant_fitness
        
        return best_solution