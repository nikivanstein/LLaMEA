import numpy as np
from scipy.optimize import differential_evolution
from sklearn.ensemble import IsolationForest

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

    def adaptive_black_box(self, func, bounds, initial_values):
        # Run Isolation Forest to handle noisy function evaluations
        isolation_forest = IsolationForest(contamination=0.35)
        isolation_forest.fit([func(x) for x in initial_values])

        # Refine the search space using Isolation Forest
        for _ in range(100):
            func_values = np.array([func(x) for x in initial_values])
            isolation_forest.partial_fit([func_values], np.zeros(self.dim))
            initial_values = np.array([x for x in initial_values if np.abs(isolation_forest.predict([func_values])[:, 0]) < 0.35])

        # Refine the search space using Genetic Algorithm
        def ga_search(bounds, initial_values):
            population = initial_values.copy()
            for _ in range(100):
                fitness = []
                for _ in range(len(population)):
                    for _ in range(self.dim):
                        population[_] = func(population[_])
                    fitness.append(self.fitness_func(population))
                population = np.array([population[i] for i in np.argsort(fitness)[:self.budget]])
            return population

        population = ga_search(bounds, initial_values)
        self.func_values = population

        # Evaluate the function at the refined search space
        self.func_values = np.array([func(x) for x in population])