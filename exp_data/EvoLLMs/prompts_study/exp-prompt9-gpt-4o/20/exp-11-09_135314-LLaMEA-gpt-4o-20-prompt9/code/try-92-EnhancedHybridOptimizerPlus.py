import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizerPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(30, budget // 10)
        self.strategy_switch = 0.25  # Switch to Differential Evolution after 25% of budget

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.initial_population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        population_size = self.initial_population_size

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Random Search with Simulated Annealing and elitism
                temperature = max(0.01, 1.0 - evals / (self.strategy_switch * self.budget))
                for i in range(population_size):
                    candidate = population[i] + np.random.normal(0, 0.3 * temperature, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evals += 1
                    if candidate_fitness < fitness[i] or np.exp((fitness[i] - candidate_fitness) / temperature) > np.random.rand():
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
                # Differential Evolution with adaptive mutation and population resizing
                scale_factor = 0.4 + 0.5 * np.random.rand()
                new_population_size = max(10, population_size - int(0.05 * population_size))
                new_population = []
                for i in range(new_population_size):
                    a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.85, mutant, population[i % population_size])
                    trial_fitness = func(trial)
                    evals += 1
                    if trial_fitness < fitness[i % population_size]:
                        new_population.append(trial)
                    else:
                        new_population.append(population[i % population_size])
                    if evals >= self.budget:
                        break
                population = np.array(new_population)
                fitness = np.apply_along_axis(func, 1, population)
                population_size = new_population_size
                idx = np.argmin(fitness)
                if fitness[idx] < best_fitness:
                    best_fitness = fitness[idx]
                    best_solution = population[idx].copy()

        return best_solution