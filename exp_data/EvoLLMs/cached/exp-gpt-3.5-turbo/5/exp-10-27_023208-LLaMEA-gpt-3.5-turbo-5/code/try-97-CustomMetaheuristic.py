import numpy as np

class CustomMetaheuristic:
    def __init__(self, budget, dim, mutation_prob=0.35):
        self.budget = budget
        self.dim = dim
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-1, 1, self.dim)
            if np.random.rand() < self.mutation_prob:
                candidate_solution += np.random.uniform(-0.1, 0.1, self.dim)
            
            candidate_fitness = func(candidate_solution)
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
        
        return best_solution

custom_metaheuristic = CustomMetaheuristic(budget=1000, dim=10)