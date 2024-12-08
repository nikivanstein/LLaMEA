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

        # Adaptive step size control
        if num_evals > 10:
            step_size = self.search_space[0] / 10
            for i in range(1, len(self.search_space)):
                if random.random() < 0.45:
                    step_size *= 0.9
                else:
                    step_size *= 1.1
            self.search_space = [x * step_size for x in self.search_space]

        return best_func

class AdaptiveMutationMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def mutate(self, individual):
        # Select a random individual from the search space
        individual_idx = random.randint(0, len(self.search_space) - 1)
        # Select a random mutation point
        mutation_idx = random.randint(0, len(self.search_space) - 1)
        # Swap the mutation point with the individual's current point
        self.search_space[individual_idx], self.search_space[mutation_idx] = self.search_space[mutation_idx], self.search_space[individual_idx]
        # Apply adaptive step size control
        if random.random() < 0.45:
            step_size = self.search_space[0] / 10
            for i in range(1, len(self.search_space)):
                if random.random() < 0.45:
                    step_size *= 0.9
                else:
                    step_size *= 1.1
            self.search_space = [x * step_size for x in self.search_space]

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization using Adaptive Step Size Control and Adaptive Mutation