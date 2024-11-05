import numpy as np

class EnhancedDynamicMutationEA:
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
            random_idx = np.random.choice([i for i in range(self.budget) if i != best_idx], 2, replace=False)
            mutant = self.population[random_idx[0]] + 0.5 * (self.population[random_idx[1]] - self.population[random_idx[2]])
            trial = best_solution + 0.5 * (mutant - best_solution)
            if func(trial) < fitness[best_idx]:
                self.population[best_idx] = trial
        return self.population[np.argmin([func(ind) for ind in self.population])]