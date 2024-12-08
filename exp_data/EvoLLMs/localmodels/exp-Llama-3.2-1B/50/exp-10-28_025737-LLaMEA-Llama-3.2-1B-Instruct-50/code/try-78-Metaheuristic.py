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

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.metaheuristic.budget, len(func(self.metaheuristic.search_space)))
        func_values = [func(x) for x in random.sample(self.metaheuristic.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Refine the strategy based on the probability of the best function
        if random.random() < 0.45:
            # Change the individual lines of the selected solution
            best_func = np.random.choice(self.metaheuristic.search_space, p=[0.3, 0.7])
        else:
            # Keep the individual lines of the selected solution the same
            pass

        # Update the search space
        self.metaheuristic.search_space = [x for x in self.metaheuristic.search_space if x not in best_func]

        return best_func

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the strategy based on the probability of the best function value
# 