import numpy as np

class EnhancedOptimizationAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full(dim, 0.5)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            
            mutation_probs = np.clip(np.exp(np.random.normal(0.5, 0.1, self.dim)), 0.1, 0.9)
            mutation_strategies = np.random.choice([np.random.standard_normal(self.budget), np.random.uniform(-1, 1, self.budget)], self.dim, p=[0.5, 0.5])
            
            for i in range(self.dim):
                population[:, i] = best_individual[i] + mutation_probs[i] * mutation_strategies[i]
            
            fitness = np.array([func(individual) for individual in population])
        
        return best_individual