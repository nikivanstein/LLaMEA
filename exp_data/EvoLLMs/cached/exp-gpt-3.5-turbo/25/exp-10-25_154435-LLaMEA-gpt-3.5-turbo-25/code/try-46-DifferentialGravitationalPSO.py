import numpy as np

class DifferentialGravitationalPSO:
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
                    else:
                        # Particle Swarm Optimization
                        w = 0.5  # Inertia weight
                        c1 = 1.5  # Cognitive factor
                        c2 = 1.5  # Social factor
                        velocity = np.random.uniform(-1, 1, self.dim)
                        best_personal = np.copy(best_solution)
                        best_global = np.copy(best_solution)
                        velocity = w * velocity + c1 * np.random.rand() * (best_personal - solution) + c2 * np.random.rand() * (best_global - solution)
                        solution += velocity
                
        return best_solution