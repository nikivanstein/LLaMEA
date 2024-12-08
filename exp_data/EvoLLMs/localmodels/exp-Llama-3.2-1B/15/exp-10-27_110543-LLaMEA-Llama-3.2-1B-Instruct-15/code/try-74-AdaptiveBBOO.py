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

        def evaluate_fitness(individual):
            fitness = evaluate_budget(eval_func, individual, self.budget)
            return fitness

        def mutate(individual):
            return individual + np.random.normal(0, 1, size=self.dim)

        def crossover(parent1, parent2):
            return np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

        def selection(population):
            return np.argsort(population, axis=0)[:self.population_size]

        def evolve_population(population, mutation_rate, crossover_rate):
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1 = selection(population[i])
                parent2 = selection(population[i])
                if np.random.rand() < mutation_rate:
                    new_population[i] = mutate(parent1)
                else:
                    new_population[i] = crossover(parent1, parent2)
                fitness = evaluate_fitness(new_population[i])
                new_population[i] = new_population[i][np.argsort(population[i], axis=0)]
            return new_population

        # Select the fittest individuals
        self.population = selection(self.population)

        # Evolve the population
        for _ in range(100):
            next_population = evolve_population(self.population, 0.1, 0.3)
            self.population = next_population

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies