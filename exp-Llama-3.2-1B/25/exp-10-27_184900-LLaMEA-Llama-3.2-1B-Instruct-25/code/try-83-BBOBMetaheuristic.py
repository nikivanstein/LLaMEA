import random
import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.population = None
        self.fitness_history = None
        self.logger = None

    def __call__(self, func):
        if self.func is None:
            raise ValueError("func() must be called before calling __call__")

        # Initialize search space
        self.search_space = [(-5.0, 5.0)] * self.dim

        # Randomly select an initial point
        self.x0 = random.uniform(self.search_space)

        # Initialize population
        self.population = [self.x0]

        # Evaluate function at each point in population
        for _ in range(self.budget):
            # Evaluate function at each point in population
            func_values = [func(self.x) for self.x in self.population]

            # Select parents using tournament selection
            self.population = self.tournament_selection(func_values)

            # Crossover (recombination)
            self.population = self.recombination(self.population)

            # Mutate (perturbation)
            self.population = self.mutation(self.population)

        # Return best individual
        best_individual = self.population[0]
        best_func_value = func(best_individual)
        for individual in self.population:
            func_value = func(individual)
            if func_value < best_func_value:
                best_individual = individual
                best_func_value = func_value

        self.fitness_history = [best_func_value] * len(self.population)

        return best_individual, best_func_value

    def tournament_selection(self, func_values):
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

        return parents

    def recombination(self, population):
        # Crossover (recombination)
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = [x for x in parent1 if x not in parent2] + [x for x in parent2 if x not in parent1]
        return child

    def mutation(self, population):
        # Mutate (perturbation)
        for i in range(self.dim):
            if random.random() < 0.1:
                population[i] += random.uniform(-1.0, 1.0)

        return population

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization on BBOB test suite
# using tournament selection, recombination, and mutation to search for the optimal solution