import numpy as np
import random

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

        return best_func

class MutationMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Generate a new individual by flipping a random bit
        new_individual = self.search_space[:]

        # Update the individual with the new bit
        new_individual[new_individual.index(random.randint(0, dim-1))] = 1 - new_individual[new_individual.index(random.randint(0, dim-1))]

        # Evaluate the new individual
        new_individual_fitness = func(new_individual)

        # Update the best individual
        if new_individual_fitness > func(new_individual):
            self.search_space = new_individual

        return new_individual

class SelectionMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))

        # Select the best individual
        best_individual = max(set(func(self.search_space), key=func(self.search_space).count))

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_individual]

        return best_individual

class SelectionAndMutationMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))

        # Select the best individual
        best_individual = max(set(func(self.search_space), key=func(self.search_space).count))

        # Generate a new individual by flipping a random bit
        new_individual = self.search_space[:]

        # Update the individual with the new bit
        new_individual[new_individual.index(random.randint(0, dim-1))] = 1 - new_individual[new_individual.index(random.randint(0, dim-1))]

        # Evaluate the new individual
        new_individual_fitness = func(new_individual)

        # Update the best individual
        if new_individual_fitness > func(new_individual):
            self.search_space = new_individual

        return new_individual

# Test the NMABBO algorithm
nmabbo = SelectionAndMutationMetaheuristic(100, 10)
best_individual = nmabbo(func, 10)

# Print the best individual
print("Best individual:", best_individual)
print("Fitness:", func(best_individual))