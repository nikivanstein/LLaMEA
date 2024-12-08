import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate fitness based on function value
            fitness = np.array([func(individual) for individual in self.population])
            # Select parents based on fitness proportionate selection
            fitness_probs = fitness / np.sum(fitness)
            selection_idx = np.random.choice(np.arange(self.budget), size=2, replace=False, p=fitness_probs)
            parents = self.population[selection_idx]
            # Perform crossover and mutation
            offspring = 0.5 * (parents[0] + parents[1]) + np.random.normal(0, 1, self.dim)
            # Replace worst individual with offspring
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]