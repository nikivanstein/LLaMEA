import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]
            
            # Dynamic mutation scale based on fitness
            fitness_parent1 = func(parent1)
            fitness_parent2 = func(parent2)
            beta = np.random.uniform(0.5, 1.0, self.dim) * (fitness_parent1 / (fitness_parent1 + fitness_parent2))
            offspring = parent1 + beta * (parent2 - self.population)

            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

        return self.population[idx[0]]