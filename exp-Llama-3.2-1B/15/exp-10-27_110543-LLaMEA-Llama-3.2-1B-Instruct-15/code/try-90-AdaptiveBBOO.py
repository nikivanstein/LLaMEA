import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim, refinement_ratio=0.15):
        self.budget = budget
        self.dim = dim
        self.refinement_ratio = refinement_ratio
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
                # Refine the individual's strategy
                refined_individual = self.refine_individual(i)
                next_population[i] = refined_individual + np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        return self.population

    def refine_individual(self, index):
        # Calculate the fitness of the current individual
        fitness = self.fitnesses[index]
        # Calculate the fitness of the best individual in the population
        best_individual = self.population[np.argsort(self.fitnesses, axis=0)][0]
        # Calculate the difference between the current individual and the best individual
        difference = best_individual - self.population[index]
        # Calculate the proportion of the difference
        proportion = np.sum(difference ** 2) / np.sum((best_individual - self.population[index]) ** 2)
        # Refine the individual's strategy based on the proportion
        refined_individual = self.population[index] + proportion * (self.population[index] - best_individual)
        return refined_individual