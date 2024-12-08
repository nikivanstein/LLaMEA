import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, max(10, budget // 5))  # Adjusted population size
        self.strategy_switch = 0.25  # Switch to Differential Evolution after 25% of budget
        self.mutation_prob = 0.7  # Mutation probability factor

    def __call__(self, func):
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
                for i in range(self.population_size):
                    candidate = population[i] + np.random.normal(0, 0.3, self.dim)  # Adjusted mutation strength
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
                if evals + self.dim + 1 <= self.budget:
                    result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.dim + 2})  # Slightly increase maxfev
                    evals += result.nfev
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x
            else:
                scale_factor_range = (0.5, 0.9)  # Broaden scale factor range
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    scale_factor = np.random.uniform(*scale_factor_range)
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < self.mutation_prob, mutant, population[i])
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