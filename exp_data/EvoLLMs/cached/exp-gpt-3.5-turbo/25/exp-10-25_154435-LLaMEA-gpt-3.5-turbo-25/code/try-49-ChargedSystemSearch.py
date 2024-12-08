import numpy as np

class ChargedSystemSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        k = 1.38e-23  # Boltzmann constant
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
                    # Electrostatic forces
                    distance = np.linalg.norm(best_solution - solution)
                    force = k * (1 / distance**2)
                    direction = (solution - best_solution) / distance
                    charged_solution = best_solution + force * direction
                    charged_fitness = func(charged_solution)
                    
                    if charged_fitness < best_fitness:
                        best_solution = charged_solution
                        best_fitness = charged_fitness
        
        return best_solution