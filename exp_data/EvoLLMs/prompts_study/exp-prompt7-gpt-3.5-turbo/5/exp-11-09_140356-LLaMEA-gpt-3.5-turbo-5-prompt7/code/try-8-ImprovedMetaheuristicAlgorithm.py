import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rate = np.ones((budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]

            # Dynamic mutation strategy
            beta = np.random.uniform(0.5, 1.0, self.dim)
            offspring = parent1 + self.mutation_rate * beta * (parent2 - self.population)

            # Replace worst solution
            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

            # Update mutation rates dynamically
            successful_mutation = np.where([func(x) < func(self.population[idx_worst]) for x in offspring])
            self.mutation_rate[successful_mutation] *= 1.05
            unsuccessful_mutation = np.where([func(x) >= func(self.population[idx_worst]) for x in offspring])
            self.mutation_rate[unsuccessful_mutation] *= 0.95

        return self.population[idx[0]]