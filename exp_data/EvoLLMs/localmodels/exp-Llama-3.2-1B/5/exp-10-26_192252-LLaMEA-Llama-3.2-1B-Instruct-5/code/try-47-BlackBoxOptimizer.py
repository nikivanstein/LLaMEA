import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5
        self.budget_evals = 0

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

        # Update the budget based on the number of evaluations
        self.budget_evals += 1
        if self.budget_evals >= self.budget:
            self.budget_evals = 0
            # Update the population with the new offspring
            self.population = np.array(population) + np.random.uniform(-1, 1, size=(self.population_size, self.dim))
            # Evaluate the function for each point in the population
            for _ in range(self.budget):
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

# One-line description: 
# An evolutionary algorithm that uses a population-based approach to explore the search space, with a balance between exploration and exploitation.