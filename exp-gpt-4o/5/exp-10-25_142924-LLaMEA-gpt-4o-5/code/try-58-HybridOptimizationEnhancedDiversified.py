import numpy as np
from scipy.optimize import minimize

class HybridOptimizationEnhancedDiversified:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.60)  # Slightly more focus on differential evolution
        self.num_nm = budget - self.num_de

    def __call__(self, func):
        population_size = 15 * self.dim  # Slightly larger population for diversity
        F = 0.70  # Increased mutation factor for wider exploration
        CR = 0.85  # Slightly reduced crossover rate to increase diversity
        delta_F = 0.04  # Adjusted delta
        delta_CR = 0.02  # Adjusted delta
        epsilon = 0.002  # Lower probability for alternative mutation strategy

        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.num_de:
            for i in range(population_size):
                if np.random.rand() < epsilon:
                    idxs = np.random.choice(list(set(range(population_size)) - {i}), 6, replace=False)  # More points for diverse strategy
                    weights = np.random.dirichlet([0.8]*6)  # Adjusted weights for more balance
                    mutant = np.clip(np.dot(weights, population[idxs]), self.lower_bound, self.upper_bound)
                else:
                    idxs = np.random.choice(list(set(range(population_size)) - {i}), 4, replace=False)  # More base vectors for stronger exploration
                    a, b, c, d = population[idxs]
                    mutant = np.clip(a + F * (b - c) + 0.5 * (d - a), self.lower_bound, self.upper_bound)  # Combined mutation strategy

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    F = min(1.0, F + delta_F * 0.25)  # Slightly reduced adaptation
                    CR = min(1.0, CR + delta_CR * 0.35)
                else:
                    F = max(0.12, F - delta_F)  # Adjusted lower bound
                    CR = max(0.17, CR - delta_CR)

                if evaluations >= self.num_de:
                    break

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        if evaluations < self.budget:
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                              options={'maxiter': self.num_nm, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7})  # Tightened tolerance
            evaluations += result.nfev
            if result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution