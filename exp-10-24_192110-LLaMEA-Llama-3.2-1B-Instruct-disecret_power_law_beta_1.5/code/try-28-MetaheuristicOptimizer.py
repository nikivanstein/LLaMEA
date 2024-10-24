import numpy as np

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = None

    def __call__(self, func):
        if self.population is None:
            self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        best_func = None
        best_score = -np.inf

        for _ in range(self.budget):
            # Select a random individual from the current population
            individual = np.random.choice(self.population, self.dim)

            # Evaluate the function at the current individual
            func_value = func(individual)

            # If this individual is better than the current best, update the best individual and score
            if func_value > best_score:
                best_func = individual
                best_score = func_value

        return best_func, best_score

    def mutate(self, func, individual):
        # Randomly select a new individual within the search space
        new_individual = individual + np.random.uniform(-1.0, 1.0, self.dim)

        # Evaluate the new individual
        new_func_value = func(new_individual)

        # If the new individual is better than the current best, update the best individual and score
        if new_func_value > best_score:
            best_func = new_individual
            best_score = new_func_value

        return new_individual, new_func_value