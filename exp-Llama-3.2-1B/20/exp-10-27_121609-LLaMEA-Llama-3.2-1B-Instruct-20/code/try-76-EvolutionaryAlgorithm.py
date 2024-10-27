import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_rate, mutation_threshold):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.mutation_threshold = mutation_threshold
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

            # Select the next individual based on the budget
            selected_individuals = np.random.choice(self.population_size, self.population_size, replace=False)
            selected_individuals = selected_individuals[:int(0.2 * self.population_size)]
            selected_individuals = selected_individuals[np.argsort(self.fitnesses[selected_individuals, :])]

            # Select the next individual based on the probability of mutation
            if random.random() < self.mutation_rate:
                mutated_individuals = self.population.copy()
                mutated_individuals[random.randint(0, self.population_size - 1)] = random.uniform(bounds(mutated_individuals[random.randint(0, self.population_size - 1)]))
                mutated_individuals = mutated_individuals[:int(0.2 * self.population_size)]
                mutated_individuals = mutated_individuals[np.argsort(self.fitnesses[mutated_individuals, :])]

            # Refine the selected individual based on the mutation
            if random.random() < self.mutation_threshold:
                mutated_individuals[random.randint(0, self.population_size - 1)] = bounds(mutated_individuals[random.randint(0, self.population_size - 1)])

            # Replace the old population with the new one
            self.population = mutated_individuals.copy()

            # Evaluate the fitness of the new population
            new_individuals = self.evaluate_fitness(self.population)
            new_individuals = new_individuals[:int(0.2 * self.population_size)]
            new_individuals = new_individuals[np.argsort(self.fitnesses[new_individuals, :])]

            # Replace the old population with the new one
            self.population = new_individuals.copy()

        return self.fitnesses

    def evaluate_fitness(self, population):
        # Evaluate the fitness of each individual in the population
        fitnesses = np.zeros((len(population), self.dim))
        for i in range(len(population)):
            fitnesses[i] = np.array([objective(x) for x in population[i]])

        return fitnesses