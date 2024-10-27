import numpy as np
import random

class AdaptiveBBOOPEVO:
    def __init__(self, budget, dim, mutation_rate, bounds):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.bounds = bounds
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select new individuals based on probability
        new_individuals = []
        for _ in range(self.population_size):
            if random.random() < 0.2:
                new_individual = self.evaluate_fitness(new_individual)
                new_individuals.append(new_individual)
            else:
                new_individual = random.choice(self.population)
                new_individuals.append(new_individual)

        # Evaluate new individuals
        new_individuals = np.array(new_individuals)
        new_fitnesses = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            x = new_individuals[i]
            fitness = objective(x)
            new_fitnesses[i] = fitness

        # Select new population
        new_population = []
        for _ in range(self.population_size):
            if random.random() < 0.2:
                new_individual = new_individuals[np.random.randint(0, self.population_size)]
                new_population.append(new_individual)
            else:
                new_individual = new_individuals[np.random.randint(0, self.population_size)]
                new_population.append(new_individual)

        # Replace old population with new population
        self.population = new_population
        self.fitnesses = new_fitnesses

        return self.fitnesses

    def evaluate_fitness(self, individual):
        # Evaluate individual using the given function
        func = lambda x: individual(x)
        return func(individual)

# Create a new instance of AdaptiveBBOOPEVO
budget = 100
dim = 2
mutation_rate = 0.01
bounds = [-5.0, 5.0]
adaptive_bboopevo = AdaptiveBBOOPEVO(budget, dim, mutation_rate, bounds)

# Define a black box function
def func(x):
    return x**2 + 3*x + 2

# Evaluate the function 100 times
adaptive_bboopevo(budget, func)

# Print the fitnesses of the new population
print(adaptive_bboopevo.fitnesses)