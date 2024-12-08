import numpy as np
import random
import scipy.optimize as optimize

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.line_search_prob = 0.1

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        for i in range(self.budget):
            # Calculate fitness values
            fitness = np.array([func(x) for x in population])

            # Select best individual
            best_individual = population[np.argmin(fitness)]

            # Create new population
            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))

            # Adaptive population size
            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]

            # Line search
            if np.random.rand() < self.line_search_prob:
                new_population = self.line_search(new_population, func)

            # Replace worst individual with best individual
            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))

        # Return best individual
        return population[np.argmin(fitness)]

    def line_search(self, population, func):
        new_population = []
        for individual in population:
            x0 = individual
            cons = ({'type': 'ineq', 'fun': lambda x: x[0] - self.lower_bound},
                    {'type': 'ineq', 'fun': lambda x: x[0] - self.upper_bound})
            res = optimize.differential_evolution(func, cons, x0=x0, full_output=True, maxiter=10)
            new_population.append(res.x)
        return np.array(new_population)

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))