import numpy as np

class AMSL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def update_individual(individual, budget):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                # Select a new individual based on the budget
                if np.random.rand() < 0.2:
                    new_individual = self.evaluate_fitness(self.evaluate_individual(budget))
                else:
                    new_individual = self.population[np.random.choice(self.population_size, 1, replace=False)]
                return new_individual

        def evaluate_fitness(individual):
            updated_individual = self.update_individual(individual, self.budget)
            return updated_individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = self.evaluate_fitness(i)
                fitness = objective(individual)
                if fitness < self.fitnesses[i, individual] + 1e-6:
                    self.fitnesses[i, individual] = fitness
                    self.population[i] = individual

        return self.fitnesses

# One-line description: An adaptive multi-step learning algorithm that uses adaptive strategies to optimize the fitness function.
# Code: 