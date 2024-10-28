import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]

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

            # Replace the elite with the children
            self.elite = children

        return self.elite[0]

    def optimize_function(self, func, budget, dim):
        # Select the best individual based on the fitness function
        best_individual = self.elite[0]

        # Define the bounds for the optimization
        bounds = [(-5.0, 5.0) for _ in range(dim)]

        # Define the constraint for the optimization
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - budget})

        # Perform the optimization using minimize
        result = minimize(func, best_individual, method='SLSQP', bounds=bounds, constraints=constraints)

        # Refine the strategy based on the probability 0.45
        if random.random() < 0.45:
            best_individual = result.x
            bounds = [(-5.0, 5.0) for _ in range(dim)]

        return best_individual, bounds

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 