import numpy as np
from scipy.optimize import minimize

class AdaptiveMultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.max_population_size = min(50, budget // 8)
        self.strategy_switch = 0.3  # Switch to Differential Evolution after 30% of budget

    def __call__(self, func):
        population_size = self.max_population_size
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Adaptive Nelder-Mead with elitism
                if evals + self.dim + 1 <= self.budget:
                    result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.dim + 1})
                    evals += result.nfev
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x
                for i in range(population_size):
                    candidate = population[i] + np.random.normal(0, 0.3, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evals += 1
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness
                        if candidate_fitness < best_fitness:
                            best_fitness = candidate_fitness
                            best_solution = candidate.copy()
                    if evals >= self.budget:
                        break
            else:
                # Differential Evolution with adaptive mutation and population reduction
                population_size = max(5, int(population_size * 0.9))
                scale_factor = 0.4 + 0.4 * np.random.rand()
                for i in range(population_size):
                    a, b, c = population[np.random.choice(self.max_population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.8, mutant, population[i])
                    trial_fitness = func(trial)
                    evals += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial.copy()
                    if evals >= self.budget:
                        break

        return best_solution