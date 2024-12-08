import random
import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.population = None
        self.best_individual = None
        self.best_func_value = float('-inf')

    def __call__(self, func):
        if self.func is None:
            raise ValueError("func() must be called before calling __call__")

        # Initialize search space
        self.search_space = [(-5.0, 5.0)] * self.dim

        # Randomly select an initial point
        self.x0 = random.uniform(self.search_space)

        # Initialize population
        self.population = [self.x0]
        self.best_individual = self.population[0]
        self.best_func_value = self.evaluate_fitness(self.population[0], func)

        # Evaluate function at each point in population
        for _ in range(self.budget):
            # Evaluate function at each point in population
            func_values = [func(self.x) for self.x in self.population]

            # Select parents using tournament selection
            parents = []
            for _ in range(self.dim):
                for i in range(self.population):
                    if i == 0:
                        parents.append(self.population[i])
                    else:
                        idx = random.randint(0, len(self.population) - 1)
                        if func_values[i] > func_values[idx]:
                            parents.append(self.population[idx])

            # Crossover (recombination)
            self.population = []
            for _ in range(self.dim):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                self.population.append(parents[(parent1 + parent2) // 2])

            # Mutate (perturbation)
            for i in range(self.dim):
                if random.random() < 0.1:
                    self.population[i] += random.uniform(-1.0, 1.0)

            # Evaluate function at each point in population
            func_values = [func(self.x) for self.x in self.population]
            self.best_individual = self.population[0]
            self.best_func_value = self.evaluate_fitness(self.best_individual, func)

        return self.best_individual, self.best_func_value

    def evaluate_fitness(self, individual, func):
        return func(individual)