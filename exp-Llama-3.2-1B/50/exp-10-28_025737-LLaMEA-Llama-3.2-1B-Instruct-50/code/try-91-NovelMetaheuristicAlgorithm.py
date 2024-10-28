import random
import numpy as np
from scipy.optimize import minimize

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

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def mutate(self, individual):
        # Refine the strategy by changing 10% of the individual's values
        mutated_individual = individual.copy()
        for i in range(len(individual)):
            if random.random() < 0.1:
                mutated_individual[i] = random.uniform(self.search_space[i][0], self.search_space[i][1])
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Select the parent with the best function value
        best_parent = max(set(parent1), key=parent1.count)

        # Create a new individual by combining the best parent with a random child
        child = [x for x in parent1 if x not in best_parent]
        child.append(random.choice(parent2))
        return child

    def run(self, func):
        # Run the optimization algorithm
        result = minimize(func, self.search_space, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)
        return result.x