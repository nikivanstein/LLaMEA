import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.dim_strategies = {
            'uniform': np.linspace(-5.0, 5.0, self.dim),
            'bounded': np.linspace(-10.0, 10.0, self.dim),
            'grid_search': np.linspace(-5.0, 5.0, self.dim).reshape(-1, self.dim)
        }

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
                strategy = np.random.choice(list(self.dim_strategies.keys()))
                if strategy == 'uniform':
                    next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                elif strategy == 'bounded':
                    next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                    bounds = self.dim_strategies[strategy]
                    next_population[i] = np.clip(next_population[i], bounds[0], bounds[1])
                elif strategy == 'grid_search':
                    next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                    grid_size = self.dim_strategies[strategy].shape
                    grid_x, grid_y = np.mgrid[self.dim_strategies[strategy][0]:self.dim_strategies[strategy][1], self.dim_strategies[strategy][0]:self.dim_strategies[strategy][1]]
                    next_population[i] = np.array([grid_x[i, j] + np.random.normal(0, 1, size=self.dim) for j in range(self.dim)])
            self.population[i] = next_population[i]

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies