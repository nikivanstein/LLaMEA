import numpy as np

class DynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            mutation_strength = np.random.uniform(0, 1)
            candidate_solution = best_solution + mutation_strength * np.random.standard_normal(self.dim)
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            
            self.mutation_rate = 0.9 * self.mutation_rate + 0.1 * abs(best_fitness - candidate_fitness)
            
            # Incorporating local search based on gradient
            if np.random.rand() < 0.1:
                gradient = np.gradient(candidate_solution)
                candidate_solution -= 0.01 * gradient  # Adjust step size for local search
                
                candidate_fitness = func(candidate_solution)
                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
            
        return best_solution