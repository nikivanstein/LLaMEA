import numpy as np

class ModifiedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.scale_factor = 0.5
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        for _ in range(self.budget):
            target_idx = np.random.randint(self.budget)
            base_idx1, base_idx2, base_idx3 = np.random.choice(np.delete(np.arange(self.budget), target_idx), size=3, replace=False)
            mutant_vector = population[base_idx1] + self.scale_factor * (population[base_idx2] - population[base_idx3])
            mutated_individual = population[target_idx] + np.random.standard_cauchy(self.dim) * mutant_vector
            if func(mutated_individual) < func(population[target_idx]):
                population[target_idx] = mutated_individual
                if func(mutated_individual) < func(best_solution):
                    best_solution = mutated_individual
        
        return best_solution