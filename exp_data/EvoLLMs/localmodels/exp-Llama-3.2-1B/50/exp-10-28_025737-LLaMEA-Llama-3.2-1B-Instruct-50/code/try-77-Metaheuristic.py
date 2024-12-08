import random
import numpy as np
import copy

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
        if random.random() < 0.45:
            # Select a new individual based on the search space
            new_individual = copy.deepcopy(self.search_space)
            if random.random() < 0.2:
                # Add a noise element to the new individual
                new_individual.append(np.random.uniform(-5.0, 5.0))
            elif random.random() < 0.3:
                # Change the upper bound of the new individual
                new_individual.append(max(new_individual))
            else:
                # Change the lower bound of the new individual
                new_individual.append(min(new_individual))
            return new_individual

class MutationExp:
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
        if random.random() < 0.45:
            # Select a new individual based on the search space
            new_individual = copy.deepcopy(self.search_space)
            if random.random() < 0.2:
                # Add a noise element to the new individual
                new_individual.append(np.random.uniform(-5.0, 5.0))
            elif random.random() < 0.3:
                # Change the upper bound of the new individual
                new_individual.append(max(new_individual))
            else:
                # Change the lower bound of the new individual
                new_individual.append(min(new_individual))
            return new_individual

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)
        self.mutation_exp = MutationExp(budget, dim)

    def __call__(self, func):
        # Optimize the function using the metaheuristic
        return self.metaheuristic(func)

# Example usage
if __name__ == "__main__":
    # Create a new optimizer
    optimizer = BlackBoxOptimizer(100, 10)

    # Optimize a function
    func = lambda x: x**2
    best_func = optimizer(func)
    print("Best function:", best_func)