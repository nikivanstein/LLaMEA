import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(30, budget // 12)
        self.max_population_size = min(50, budget // 8)
        self.strategy_switch_ratio = 0.3  # Switch to Differential Evolution after 30% of budget

    def __call__(self, func):
        # Initialize population
        population_size = self.initial_population_size
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch_ratio * self.budget:
                # Random Search with Nelder-Mead and elitism
                for i in range(population_size):
                    candidate = population[i] + np.random.normal(0, 0.1, self.dim)
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
                    result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.dim + 1})
                    evals += result.nfev
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x
            else:
                # Differential Evolution with adaptive mutation and dynamic population size
                population_size = min(self.max_population_size, population_size + 1)
                scale_factor = 0.5 + 0.2 * np.random.rand()
                for i in range(population_size):
                    a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
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