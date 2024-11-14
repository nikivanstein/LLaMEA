import numpy as np
from scipy.optimize import minimize

class EnhancedOppositionHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.3  # Switch to Differential Evolution after 30% of budget

    def __call__(self, func):
        # Initialize population with opposition-based learning
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        opp_population = self.lower_bound + self.upper_bound - population
        combined_population = np.vstack((population, opp_population))
        combined_fitness = np.apply_along_axis(func, 1, combined_population)
        evals = 2 * self.population_size
        indices = np.argsort(combined_fitness)[:self.population_size]
        population = combined_population[indices]
        fitness = combined_fitness[indices]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Random Search with Nelder-Mead and elitism
                for i in range(self.population_size):
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
                # Differential Evolution with adaptive mutation
                scale_factor = 0.6 + 0.2 * np.random.rand()  # Modified scale factor
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.8, mutant, population[i])  # Adjusted crossover probability
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