import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim, strategy):
        self.budget = budget
        self.dim = dim
        self.strategy = strategy
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.tolerance = 0.15
        self.max_iter = 1000

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
        for _ in range(self.max_iter):
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                if np.random.rand() < self.tolerance:
                    new_individual = self.strategy(self.population[i], eval_func)
                else:
                    new_individual = self.population[i]
                fitness = evaluate_budget(eval_func, new_individual, self.budget)
                new_population[i] = new_individual
                self.population[i] = new_population[i]

        return self.population

    def strategy(self, individual, func):
        # Refine the strategy based on the fitness of the individual
        # This can be done by changing the lines of the selected solution to refine its strategy
        # For example, you can use a linear combination of the two lines to refine the strategy
        return individual + 0.7 * np.random.normal(0, 1, size=self.dim)

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies