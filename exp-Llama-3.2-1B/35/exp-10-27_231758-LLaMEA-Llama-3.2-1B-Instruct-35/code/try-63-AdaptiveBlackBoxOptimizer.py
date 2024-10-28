import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_black_box(self, func, num_evals=1000, alpha=0.1):
        # Initialize population with random values
        population = np.random.uniform(-5.0, 5.0, size=(num_evals, self.dim))

        # Evolve population using adaptive black box optimization
        for _ in range(100):  # Run for 100 generations
            # Evaluate function at each individual in population
            func_values = np.array([func(ind) for ind in population])

            # Select fittest individuals
            idx = np.argmin(np.abs(func_values))
            fittest_individuals = population[:idx]

            # Create new population by perturbing fittest individuals
            new_population = np.array([func(ind) for ind in fittest_individuals] + [func(np.random.uniform(-5.0, 5.0)) for _ in range(num_evals - len(fittest_individuals))])

            # Replace least fit individuals with new ones
            population = np.sort(new_population)[:idx] + fittest_individuals

        # Evaluate function at the best individual in population
        func_values = np.array([func(ind) for ind in population])
        best_individual = population[np.argmin(np.abs(func_values))]
        best_func_value = func(best_individual)

        # Update population with best individual and its function value
        population = np.vstack((population, [best_individual, best_func_value]))

        # Return best individual and its function value
        return best_individual, best_func_value

# Description: Adaptive Black Box Optimization Algorithm
# Code: 