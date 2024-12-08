import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func, tuning_params):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = differential_evolution(func, [(-5.0, 5.0) for _ in range(self.population_size)], bounds=[(-5.0, 5.0) for _ in range(self.dim)], initial=self.population, tol=1e-3, maxiter=100, x0=self.population)
            if np.any(func_evals.fun!= func_evals.x):
                func_evals = differential_evolution(func, [(-5.0, 5.0) for _ in range(self.population_size)], bounds=[(-5.0, 5.0) for _ in range(self.dim)], initial=self.population, tol=1e-3, maxiter=100, x0=self.population)
            if np.all(func_evals.fun == func_evals.x):
                break
            num_evaluations += 1
        return func_evals

    def adaptive_sampling_tuned(self, func, tuning_params):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = differential_evolution(func, [(-5.0, 5.0) for _ in range(self.population_size)], bounds=[(-5.0, 5.0) for _ in range(self.dim)], initial=self.population, tol=1e-3, maxiter=100, x0=self.population, n_iter=10)
            if np.any(func_evals.fun!= func_evals.x):
                func_evals = differential_evolution(func, [(-5.0, 5.0) for _ in range(self.population_size)], bounds=[(-5.0, 5.0) for _ in range(self.dim)], initial=self.population, tol=1e-3, maxiter=100, x0=self.population, n_iter=10)
            if np.all(func_evals.fun == func_evals.x):
                break
            num_evaluations += 1
        return func_evals