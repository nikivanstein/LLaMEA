import numpy as np
from scipy.optimize import minimize

class HybridOptimizationEnhancedStochastic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.5)
        self.num_sgd = budget - self.num_de

    def __call__(self, func):
        population_size = 12 * self.dim
        F = 0.7
        CR = 0.8
        delta_F = 0.02
        delta_CR = 0.02
        epsilon = 0.001

        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        def chaotic_map(x, iterations=1):
            for _ in range(iterations):
                x = 4 * x * (1 - x)
            return x

        while evaluations < self.num_de:
            for i in range(population_size):
                if np.random.rand() < epsilon:
                    idxs = np.random.choice(list(set(range(population_size)) - {i}), 5, replace=False)
                    weights = np.random.dirichlet([1]*5)
                    mutant = np.clip(np.dot(weights, population[idxs]), self.lower_bound, self.upper_bound)
                else:
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
                    F = chaotic_map(F + delta_F * 0.3)
                    CR = chaotic_map(CR + delta_CR * 0.4)
                else:
                    F = chaotic_map(F - delta_F)
                    CR = chaotic_map(CR - delta_CR)

                if evaluations >= self.num_de:
                    break

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        if evaluations < self.budget:
            learning_rate = 0.01
            for _ in range(self.num_sgd):
                grad = np.gradient(np.array([func(best_solution + np.random.normal(0, 0.1, self.dim)) for _ in range(3)]))
                best_solution -= learning_rate * grad
                np.clip(best_solution, self.lower_bound, self.upper_bound, out=best_solution)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                              options={'maxiter': self.num_sgd, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            evaluations += result.nfev
            if result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution