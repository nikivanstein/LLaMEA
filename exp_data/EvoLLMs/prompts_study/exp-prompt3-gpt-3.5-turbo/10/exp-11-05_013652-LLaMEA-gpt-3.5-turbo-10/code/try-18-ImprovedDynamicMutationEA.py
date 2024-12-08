import numpy as np

class ImprovedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

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
            else:
                random_idx = np.random.choice([idx for idx in range(self.budget) if idx != best_idx])
                self.population[best_idx] = self.crossover(self.population[best_idx], self.population[random_idx])
        return self.population[np.argmin([func(ind) for ind in self.population])]