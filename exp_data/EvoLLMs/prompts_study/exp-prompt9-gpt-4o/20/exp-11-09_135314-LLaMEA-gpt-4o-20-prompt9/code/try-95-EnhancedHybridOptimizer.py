import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, budget // 8)  # Increased population size
        self.strategy_switch = 0.3  # Switch to new strategy after 30% of budget

    def adaptive_random_search(self, func, population, fitness, evals):
        for i in range(self.population_size):
            candidate = population[i] + np.random.normal(0, 0.3, self.dim)  # Adjusted variance
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            evals += 1
            if candidate_fitness < fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness
        return evals

    def cma_strategy(self, func, best_solution, evals):
        if evals + self.dim + 1 <= self.budget:
            result = minimize(func, best_solution, method='Powell', options={'maxfev': self.dim + 1})  # Changed to Powell
            evals += result.nfev
        return evals, result

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
                # Adaptive Random Search with elitism
                evals = self.adaptive_random_search(func, population, fitness, evals)
                if evals >= self.budget:
                    break
                evals, result = self.cma_strategy(func, best_solution, evals)
                if result.fun < best_fitness:
                    best_fitness = result.fun
                    best_solution = result.x
            else:
                # Differential Evolution with adaptive mutation and crossover
                scale_factor = 0.6 + 0.2 * np.random.rand()  # Slight adjustment
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.85, mutant, population[i])  # Adjusted crossover rate
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