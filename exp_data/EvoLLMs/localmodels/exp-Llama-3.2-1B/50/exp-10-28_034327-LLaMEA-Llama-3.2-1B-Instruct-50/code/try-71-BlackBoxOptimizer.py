import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        def evaluate_fitness(individual):
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
            else:
                # If the point is not within bounds, return the best point found so far
                return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

        def iterated_permutation(individual, budget):
            best_individual = individual
            for _ in range(budget):
                new_individual = individual
                for i in range(self.dim):
                    if np.random.rand() < 0.45:
                        new_individual[i] += np.random.uniform(-5.0, 5.0)
                new_individual = np.clip(new_individual, -5.0, 5.0)
                if evaluate_fitness(new_individual) > evaluate_fitness(best_individual):
                    best_individual = new_individual
            return best_individual

        return iterated_permutation

# Example usage:
optimizer = BlackBoxOptimizer(100, 2)
func = lambda x: np.sin(x)
best_individual = optimizer(func)
print(best_individual)