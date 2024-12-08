import numpy as np
import random

class AdaptiveMutationGA:
    def __init__(self, budget, dim, mutation_prob=0.2, mutation_size=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_prob = mutation_prob
        self.mutation_size = mutation_size

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            mutated_individual = individual.copy()
            for _ in range(random.randint(0, self.budget)):
                if random.random() < self.mutation_prob:
                    mutated_individual[random.randint(0, self.dim-1)] += random.uniform(-self.mutation_size, self.mutation_size)
            return mutated_individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutate(x)

        return self.fitnesses

    def evaluate_fitness(self, individual):
        updated_individual = individual.copy()
        for _ in range(self.budget):
            updated_individual = self.f(individual, updated_individual)
        return updated_individual

# Example usage:
ga = AdaptiveMutationGA(budget=100, dim=10)
func = lambda x: x**2
selected_solution = ga(func)
print(selected_solution)