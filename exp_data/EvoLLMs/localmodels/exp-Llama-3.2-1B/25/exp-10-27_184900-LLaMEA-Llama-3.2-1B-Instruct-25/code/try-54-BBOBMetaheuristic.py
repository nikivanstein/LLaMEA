import random
import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None

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
                if random.random() < 0.25:
                    self.population[i] += random.uniform(-1.0, 1.0)

        # Return best individual
        best_individual = self.population[0]
        best_func_value = func(best_individual)
        for individual in self.population:
            func_value = func(individual)
            if func_value < best_func_value:
                best_individual = individual
                best_func_value = func_value

        return best_individual, best_func_value

# Novel Metaheuristic Algorithm for Black Box Optimization on BBOB Test Suite
# using tournament selection, recombination, and mutation to search for the optimal solution
# with an improved mutation strategy to reduce the impact of local optima
# and a more efficient crossover strategy to improve the diversity of the population