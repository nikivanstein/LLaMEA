import numpy as np
from scipy.optimize import minimize

class HybridOptimizationEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.65)  # Adjusted DE budget allocation for better exploration
        self.num_nm = budget - self.num_de

    def __call__(self, func):
        # Adaptive Differential Evolution (DE) parameters
        population_size = 10 * self.dim  # Reduced population size for faster convergence
        F = 0.65  # Slightly decreased differential weight
        CR = 0.75  # Reduced crossover probability for exploration
        delta_F = 0.05  # Adaptation step for F
        delta_CR = 0.03  # Adaptation step for CR

        # Initialize DE population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # DE loop with dynamic mutation strategy
        while evaluations < self.num_de:
            for i in range(population_size):
                idxs = np.random.choice(list(set(range(population_size)) - {i}), 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    F = min(1.0, F + delta_F * 0.5)
                    CR = min(1.0, CR + delta_CR * 0.5)
                else:
                    F = max(0.3, F - delta_F)  # Allowing F to decrease more for exploration
                    CR = max(0.3, CR - delta_CR)  # Allowing CR to decrease more for exploration

                if evaluations >= self.num_de:
                    break

        # Take the best solution found by DE
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        # Nelder-Mead optimization for exploitation
        if evaluations < self.budget:
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                              options={'maxiter': self.num_nm, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            evaluations += result.nfev
            if result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution