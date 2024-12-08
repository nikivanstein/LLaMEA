import numpy as np

class DynamicMutationDE:
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
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
            candidate_idx = np.random.choice([i for i in range(self.budget) if i != best_idx], size=2, replace=False)
            candidate_solution = self.population[candidate_idx[0]] - self.population[candidate_idx[1]]
            candidate_mutation = np.random.randn(self.dim) * mutation_rate
            candidate_new_solution = best_solution + candidate_mutation
            if func(candidate_new_solution) < fitness[best_idx]:
                self.population[best_idx] = candidate_new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]