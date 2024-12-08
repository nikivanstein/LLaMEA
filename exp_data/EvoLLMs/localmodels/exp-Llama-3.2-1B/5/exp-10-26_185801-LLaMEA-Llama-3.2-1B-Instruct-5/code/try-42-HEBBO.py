import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        # Select a random parent from the current population
        parent1, parent2 = np.random.choice(self.search_space, size=2, replace=False)

        # Select a random mutation rate
        mutation_rate = np.random.uniform(0.0, 0.1, size=self.dim)

        # Apply mutation to the individual
        mutated_individual = individual + mutation_rate * (parent1 - individual)

        # Ensure the mutated individual stays within the search space
        mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

        return mutated_individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = np.random.randint(1, self.dim)

        # Create the offspring by combining the parents
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        return offspring

    def evaluate_fitness(self, individual, budget):
        # Evaluate the fitness of the individual using the function
        func_value = self.__call__(individual)
        return func_value