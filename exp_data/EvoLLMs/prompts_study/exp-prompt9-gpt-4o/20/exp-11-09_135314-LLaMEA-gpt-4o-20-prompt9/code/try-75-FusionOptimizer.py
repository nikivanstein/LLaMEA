import numpy as np
from scipy.optimize import minimize

class FusionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(35, budget // 9)  # Slightly larger population
        self.strategy_switch = 0.25  # Switch strategy after 25% of budget

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Adaptive Nelder-Mead with randomized restart
                for i in range(self.population_size):
                    candidate = population[i] + np.random.normal(0, 0.15, self.dim)
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
                if evals + self.dim + 2 <= self.budget:
                    result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.dim + 2})
                    evals += result.nfev
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x
            else:
                # Enhanced Differential Evolution with leader selection
                scale_factor = 0.6 + 0.2 * np.random.rand()
                for i in range(self.population_size):
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    leader = population[np.argmin(fitness[indices])]  # Select better leader
                    mutant = leader + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.85, mutant, population[i])
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