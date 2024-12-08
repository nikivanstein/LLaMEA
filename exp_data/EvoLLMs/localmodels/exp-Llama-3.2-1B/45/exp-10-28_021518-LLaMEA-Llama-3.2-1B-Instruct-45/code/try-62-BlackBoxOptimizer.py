import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.adaptation_rate = 0.01
        self.adaptation_threshold = 0.1
        self.update_population()

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        def fitness_func(x):
            return evaluate_func(x)

        while len(self.elite) < self.elite_size:
            # Selection
            fitness_values = [fitness_func(x) for x in self.population]
            indices = np.argsort(fitness_values)[:self.population_size]
            self.elite = [self.population[i] for i in indices]

            # Crossover
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(self.elite, 2)
                child = (parent1 + parent2) / 2
                children.append(child)

            # Mutation
            for child in children:
                if random.random() < 0.1:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)

            # Adaptation
            if random.random() < self.adaptation_rate:
                new_individual = self.evaluate_fitness(child)
                if new_individual is not None and new_individual < self.elite[0]:
                    self.elite[0] = new_individual

            # Replace the elite with the children
            self.elite = children

        return self.elite[0]

    def update_population(self):
        # Update the elite with the best individual
        new_individual = self.elite[0]
        new_fitness = self.f(new_individual, self.logger)
        if new_fitness is not None and new_fitness < self.elite[0]:
            self.elite[0] = new_individual
            self.update_population()

        # Update the population with the best individual
        best_individual = self.elite[0]
        best_fitness = self.f(best_individual, self.logger)
        if best_fitness is not None and best_fitness < self.elite[0]:
            self.elite = [best_individual]
            self.update_population()

    def f(self, individual, logger):
        # Evaluate the fitness of the individual
        func = individual
        updated_individual = self.budget(func, logger)
        if updated_individual is not None:
            return self.f(updated_individual, logger)
        else:
            return self.elite[0]