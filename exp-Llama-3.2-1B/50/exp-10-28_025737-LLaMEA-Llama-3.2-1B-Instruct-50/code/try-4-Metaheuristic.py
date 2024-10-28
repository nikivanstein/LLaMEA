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
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Refine the strategy by changing the individual lines of the selected solution
        for _ in range(5):
            # Select a random individual from the search space
            individual = random.choice(self.search_space)

            # Apply a mutation to the individual
            mutated_individual = individual.copy()
            if random.random() < 0.45:  # Change the individual line
                mutated_individual[0] = random.uniform(-5.0, 5.0)

            # Evaluate the function using the mutated individual
            new_func_value = func(mutated_individual)

            # Update the best function value
            if new_func_value > best_func[0]:
                best_func = [new_func_value]

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

# Initialize the algorithm
algorithm = NovelMetaheuristicAlgorithm(100, 10)
# Run the algorithm
best_func = algorithm(func)
# Print the result
print(best_func)