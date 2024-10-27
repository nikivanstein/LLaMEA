import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

    def __call__(self, func):
        def func_eval(x):
            return func(x)

        # Initialize the population with random points in the search space
        x = np.random.uniform(-5.0, 5.0, self.dim)
        population = [x] * self.population_size

        # Evaluate the function for each point in the population
        for _ in range(self.budget):
            # Select the fittest points to reproduce
            fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

            # Create new offspring by crossover and mutation
            offspring = []
            for i in range(self.population_size // 2):
                parent1, parent2 = random.sample(fittest_points, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child += random.uniform(-5.0, 5.0)
                offspring.append(child)

            # Replace the worst points in the population with the new offspring
            population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Select the fittest points to reproduce
        fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

        # Create new offspring by crossover and mutation
        offspring = []
        for i in range(self.population_size // 2):
            parent1, parent2 = random.sample(fittest_points, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child += random.uniform(-5.0, 5.0)
            offspring.append(child)

        # Replace the worst points in the population with the new offspring
        population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Create a new population with the selected solution
        new_population = [x for x in population if func_eval(x) < func_eval(self.evaluate_fitness(x))]
        new_population.extend([x for x in population if func_eval(x) >= func_eval(self.evaluate_fitness(x))])

        return new_population

    def evaluate_fitness(self, func):
        return func(self.evaluate_individual(func))

    def evaluate_individual(self, func, logger):
        # Refine the strategy by changing individual lines
        if random.random() < 0.05:
            new_individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
        else:
            new_individual = self.evaluate_fitness(new_individual, logger)

        # Evaluate the new individual
        new_individual = func(new_individual)

        # Update the logger with the new individual's fitness
        logger.info(f"Individual: {new_individual}, Fitness: {new_individual(self.evaluate_fitness(new_individual))}")

        return new_individual