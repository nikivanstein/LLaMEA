import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.mutation_rate = 0.1

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
                mutation_rate = self.mutation_rate
                if np.random.rand() < mutation_rate:
                    mutation_rate = 1 - mutation_rate
                next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        # Refine the strategy based on the fitness scores
        for i in range(self.population_size):
            fitness = self.fitnesses[i]
            if fitness < 0.75:
                mutation_rate = 0.1
            elif fitness < 0.9:
                mutation_rate = 0.2
            else:
                mutation_rate = 0.3
            next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
            fitness = evaluate_budget(eval_func, next_population[i], self.budget)
            next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
            self.population[i] = next_population[i]

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies with Adaptive Mutation Rate