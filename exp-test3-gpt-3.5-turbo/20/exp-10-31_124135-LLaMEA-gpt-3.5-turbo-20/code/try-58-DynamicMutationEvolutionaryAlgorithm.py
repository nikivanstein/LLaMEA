import numpy as np

class DynamicMutationEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_strength = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Dynamic mutation strength based on best individual's fitness
            mutation_strength = self.mutation_strength / (1 + fitness[best_idx])
            mutated_population = population + np.random.normal(0, mutation_strength, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget
        
        return best_solution