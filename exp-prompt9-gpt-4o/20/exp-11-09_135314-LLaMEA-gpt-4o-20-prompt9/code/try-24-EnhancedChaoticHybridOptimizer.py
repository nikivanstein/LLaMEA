import numpy as np
from scipy.optimize import minimize

class EnhancedChaoticHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.3  # Adjusted switch to Differential Evolution after 30% of budget
        self.chaos_map = self.logistic_map  # Using logistic map for chaotic initialization

    def logistic_map(self, x, r=3.99):
        return r * x * (1 - x)

    def initialize_population(self):
        chaotic_vals = self.chaos_map(np.random.rand(self.population_size, self.dim))
        return self.lower_bound + (self.upper_bound - self.lower_bound) * chaotic_vals

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                for i in range(self.population_size):
                    candidate = population[i] + np.random.normal(0, 0.2, self.dim)
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
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    scale_factor = 0.5 + 0.3 * np.random.rand()
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    crossover_rate = 0.7 + 0.2 * np.random.rand()  # Adaptive crossover
                    trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
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