import numpy as np
from scipy.optimize import differential_evolution

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

    def adaptive_search(self, func, bounds, initial_point, mutation_rate):
        # Initialize the population with random points within the search space
        population = initial_point + np.random.uniform(-bounds, bounds, size=(self.budget, self.dim))
        
        # Evaluate the fitness of each point and select the fittest ones
        fitness = np.array([func(point) for point in population])
        population = population[np.argsort(fitness)]
        
        # Perform genetic algorithm iterations
        for _ in range(100):
            # Select parents using tournament selection
            parents = np.array([population[0], population[np.random.choice(len(population), size=1, replace=False)]]).T
            # Apply mutation
            for i in range(self.budget):
                idx = np.random.randint(0, len(parents))
                mutated_point = parents[idx, 0] + np.random.uniform(-bounds[i], bounds[i], size=self.dim)
                if np.random.rand() < mutation_rate:
                    mutated_point = np.clip(mutated_point, bounds[i], None)
                parents[idx, 0] = mutated_point
            # Evaluate the fitness of each point and select the fittest ones
            fitness = np.array([func(point) for point in parents])
            population = population[np.argsort(fitness)]
        
        # Return the fittest point as the solution
        return population[0, 0]