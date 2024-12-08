import numpy as np

class HybridDynamicMutationSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation_rate = np.random.uniform(0.01, 0.1)
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = best_solution + mutation
            
            # Simulated Annealing
            temperature = 1.0
            acceptance_probability = np.exp((func(new_solution) - func(best_solution)) / temperature)
            if func(new_solution) < func(best_solution) or np.random.rand() < acceptance_probability:
                self.population[best_idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]