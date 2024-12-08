import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.search_space = np.linspace(-5.0, 5.0, 10)
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Refine the strategy using a genetic algorithm
        while True:
            # Generate a new population
            new_population = self.generate_new_population()

            # Evaluate the new population
            new_population_evals = self.evaluate_population(new_population)

            # Select the fittest individuals
            fittest_individuals = self.select_fittest_individuals(new_population_evals, self.population_size)

            # Mutate the fittest individuals
            mutated_individuals = self.mutate_individuals(fittest_individuals, self.mutation_rate)

            # Replace the old population with the new one
            self.population_size = len(mutated_individuals)
            self.func_values = mutated_individuals
            self.func_evals = len(mutated_individuals)
            self.func_values = np.zeros(self.dim)

            # Check for convergence
            if self.func_evals / self.func_evals < 0.35:
                break

        # Return the fittest individual
        return fittest_individuals[0]

    def generate_new_population(self):
        new_population = []
        for _ in range(self.population_size):
            individual = np.random.choice(self.search_space, size=self.dim)
            new_population.append(individual)
        return new_population

    def evaluate_population(self, population):
        return np.mean(np.abs(np.array(population) - np.array(self.func_values)))

    def select_fittest_individuals(self, population_evals, population_size):
        # Simple selection strategy: select the top k individuals
        return np.random.choice(population_size, size=population_size, p=population_evals / population_size)

    def mutate_individuals(self, individuals, mutation_rate):
        mutated_individuals = []
        for individual in individuals:
            if random.random() < mutation_rate:
                idx = random.randint(0, self.dim - 1)
                individual[idx] = np.random.uniform(-5.0, 5.0)
            mutated_individuals.append(individual)
        return mutated_individuals