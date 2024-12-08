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

        def select_parents(population, fitnesses):
            return np.random.choice(population, size=self.population_size, replace=True, p=fitnesses / np.sum(fitnesses))

        def mutate(population, mutation_rate):
            return np.random.normal(0, 1, size=population.shape)

        def crossover(parent1, parent2):
            return np.concatenate((parent1[:int(self.population_size/2)], parent2[int(self.population_size/2):]))

        def evolve(population, mutation_rate, crossover_rate):
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = select_parents(population, self.fitnesses[i])
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                new_population[i] = child
            return new_population

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        for _ in range(self.budget):
            fitnesses = evaluate_budget(eval_func, self.population, self.budget)
            self.fitnesses = fitnesses
            self.population = evolve(self.population, 0.1, 0.1)

        # Select the fittest individuals
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            next_population = evolve(self.population, 0.1, 0.1)
            self.population = next_population

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies