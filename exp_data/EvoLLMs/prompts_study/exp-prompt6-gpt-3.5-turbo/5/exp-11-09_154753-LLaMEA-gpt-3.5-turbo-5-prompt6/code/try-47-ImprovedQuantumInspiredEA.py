import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        F = 0.5  # Scaling factor for differential evolution
        for _ in range(self.budget):
            # Evaluate fitness based on function value
            fitness = [func(individual) for individual in self.population]
            # Select parents using differential evolution strategy
            parents = self.population[np.argsort(fitness)[:2]]
            mutant = parents[0] + F * (parents[1] - parents[0])  # Mutation
            crossover_prop = np.random.rand(self.dim) < 0.8  # Crossover probability
            offspring = parents[0].copy()
            offspring[crossover_prop] = mutant[crossover_prop]  # Crossover
            # Replace worst individual with offspring
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]