import numpy as np
from scipy.optimize import minimize

class HybridOptimizationEnhancedSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.55)
        self.num_nm = budget - self.num_de

    def __call__(self, func):
        population_size = 14 * self.dim
        F = 0.65
        CR = 0.9
        delta_F = 0.05
        delta_CR = 0.03
        epsilon = 0.001
        T_init = 1.0  # Initial temperature for simulated annealing
        cooling_rate = 0.99  # Cooling rate for temperature

        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        temperature = T_init

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
                accept = trial_fitness < fitness[i] or \
                         np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature)
                if accept:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    F = min(1.0, F + delta_F * 0.3)
                    CR = min(1.0, CR + delta_CR * 0.4)
                else:
                    F = max(0.1, F - delta_F)
                    CR = max(0.15, CR - delta_CR)

                if evaluations >= self.num_de:
                    break

            temperature *= cooling_rate

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        if evaluations < self.budget:
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                              options={'maxiter': self.num_nm, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            evaluations += result.nfev
            if result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution