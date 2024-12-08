import numpy as np

class DynamicMutationMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rate = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]

            # Dynamic mutation strategy
            beta = np.random.uniform(0.5, 1.0, self.dim) * self.mutation_rate
            offspring = parent1 + beta * (parent2 - self.population)

            # Replace worst solution
            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

            # Update mutation rate dynamically
            self.mutation_rate = min(1.0, self.mutation_rate + 0.005)

        return self.population[idx[0]]