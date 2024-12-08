import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def mutate(individual):
            mutated_individual = individual.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    mutated_individual[i] += np.random.normal(0, 1)
            return mutated_individual

        def crossover(parent1, parent2):
            child = parent1.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    child[i] = parent2[i]
            return child

        def adapt(individual):
            if np.random.rand() < 0.15:
                individual = mutate(individual)
            if np.random.rand() < 0.15:
                individual = crossover(individual, individual)
            return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = evaluate_budget(eval_func, self.population[i], self.budget)
                self.fitnesses[i] = fitness
                self.population_history.append(self.population[i])

        # Select the fittest individuals
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            next_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                next_population[i] = adapt(self.population[i])
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies