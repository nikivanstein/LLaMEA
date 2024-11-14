import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents = self.population[np.argsort(fitness)[:3]]  # Select 3 parents
            mutant = parents[0] + 0.5 * (parents[1] - parents[2])  # Mutant creation
            crossover_prob = np.random.rand(self.dim) < 0.9  # Crossover probability
            offspring = np.where(crossover_prob, mutant, self.population[np.argsort(fitness)[0]])  # Crossover
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]