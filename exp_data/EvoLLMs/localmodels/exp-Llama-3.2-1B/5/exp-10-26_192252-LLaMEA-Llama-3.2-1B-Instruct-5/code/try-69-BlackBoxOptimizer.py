import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5
        self.budget_evaluations = 0

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

        # Update the budget evaluations
        self.budget_evaluations += 1
        if self.budget_evaluations >= self.budget:
            self.budget_evaluations = 0
            # Update the population using a new strategy
            self.population = self.update_population(func)

        return population

    def update_population(self, func):
        # Define the new strategy
        strategy = "explore_exploit"
        if random.random() < 0.5:
            # Explore the search space
            new_individual = self.evaluate_fitness(self.population)
            # Update the population with the new individual
            self.population = [new_individual] * self.population_size
        else:
            # Exploit the current population
            new_individual = self.population[0]
            for _ in range(self.population_size):
                # Select the fittest points to reproduce
                fittest_points = sorted(self.population, key=func_eval, reverse=True)[:self.population_size // 2]

                # Create new offspring by crossover and mutation
                offspring = []
                for i in range(self.population_size // 2):
                    parent1, parent2 = random.sample(fittest_points, 2)
                    child = (parent1 + parent2) / 2
                    if random.random() < self.mutation_rate:
                        child += random.uniform(-5.0, 5.0)
                    offspring.append(child)

                # Replace the worst points in the population with the new offspring
                self.population = [x if func_eval(x) < func_eval(p) else p for p in self.population]

        # Select the fittest points to reproduce
        fittest_points = sorted(self.population, key=func_eval, reverse=True)[:self.population_size // 2]

        # Create new offspring by crossover and mutation
        offspring = []
        for i in range(self.population_size // 2):
            parent1, parent2 = random.sample(fittest_points, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child += random.uniform(-5.0, 5.0)
            offspring.append(child)

        # Replace the worst points in the population with the new offspring
        self.population = [x if func_eval(x) < func_eval(p) else p for p in self.population]

        return self.population