import numpy as np
from scipy.optimize import minimize

class HybridOptimizationAdvanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.60)  # Adjusted DE budget allocation
        self.num_nm = budget - self.num_de

    def __call__(self, func):
        population_size = 14 * self.dim
        F = 0.65
        CR = 0.9
        delta_F = 0.07  # Slightly increased step for F adjustment
        delta_CR = 0.04  # Slightly increased step for CR adjustment
        epsilon = 0.001

        # Quasi-random initialization using Sobol sequence
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=self.dim, scramble=True)
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * sampler.random(population_size)
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.num_de:
            for i in range(population_size):
                idxs = np.random.choice(list(set(range(population_size)) - {i}), 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    F = min(1.0, F + delta_F * 0.25)  # Adjusted F adaptation
                    CR = min(1.0, CR + delta_CR * 0.3)  # Adjusted CR adaptation
                else:
                    F = max(0.1, F - delta_F * 0.5)  # More conservative decrease
                    CR = max(0.1, CR - delta_CR * 0.5)

                if evaluations >= self.num_de:
                    break

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        if evaluations < self.budget:
            result = minimize(func, best_solution, method='Powell',  # Switched to Powell for potential robustness
                              bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                              options={'maxiter': self.num_nm, 'xatol': 1e-6, 'fatol': 1e-6})
            evaluations += result.nfev
            if result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution