import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        # Refine the strategy
        if len(self.search_space) > 1:
            # Calculate the probability of selecting each point
            probabilities = [func(x) / len(self.search_space) for x in self.search_space]
            # Select the next point based on the probabilities
            new_individual = random.choices(self.search_space, weights=probabilities, k=1)[0]
        else:
            new_individual = random.choice(self.search_space)

        return new_individual

class BBOB:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        # Initialize the population
        self.population = [Metaheuristic(self.budget, self.dim) for _ in range(100)]
        for individual in self.population:
            individual.initialize()

        # Run the optimization algorithm
        while self.population[-1].best_fitness!= self.best_fitness:
            # Evaluate the function
            fitness = self.func(self.population[-1].search_space)
            # Select the next individual
            self.population[-1] = self.population[-1].__call__(self.func)

            # Refine the strategy
            if len(self.population[-1].search_space) > 1:
                # Calculate the probability of selecting each point
                probabilities = [fitness / len(self.population[-1].search_space) for _ in range(self.dim)]
                # Select the next point based on the probabilities
                new_individual = random.choices(self.population[-1].search_space, weights=probabilities, k=1)[0]
                # Update the search space
                self.population[-1].search_space = [x for x in self.population[-1].search_space if x not in new_individual]
            else:
                new_individual = random.choice(self.population[-1].search_space)

            # Add the new individual to the population
            self.population.append(Metaheuristic(self.budget, self.dim).initialize())

        # Update the best individual and fitness
        self.best_individual = self.population[0].search_space
        self.best_fitness = fitness

        # Return the best individual
        return self.best_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 