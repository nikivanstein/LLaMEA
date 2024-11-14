import numpy as np

class DynamicMutationMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]

            # Dynamic mutation strategy
            fitness_values = [func(x) for x in self.population]
            mutation_rate = np.exp(-np.std(fitness_values))
            beta = np.random.uniform(0.5, 1.0, self.dim) * mutation_rate
            offspring = parent1 + beta * (parent2 - self.population)

            # Replace worst solution
            idx_worst = np.argmax(fitness_values)
            self.population[idx_worst] = offspring

        return self.population[idx[0]]