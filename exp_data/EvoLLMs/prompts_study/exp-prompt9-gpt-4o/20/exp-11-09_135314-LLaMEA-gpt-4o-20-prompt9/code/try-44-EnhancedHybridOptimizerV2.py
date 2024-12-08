import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.2

    def simulated_annealing(self, func, x0):
        temperature = 1.0
        cooling_rate = 0.95
        current_solution = x0
        current_fitness = func(current_solution)
        evals = 0

        while temperature > 1e-3 and evals < self.budget:
            candidate = current_solution + np.random.normal(0, 0.2, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            evals += 1

            if candidate_fitness < current_fitness or \
               np.exp((current_fitness - candidate_fitness) / temperature) > np.random.rand():
                current_solution = candidate
                current_fitness = candidate_fitness

            temperature *= cooling_rate
            if evals >= self.budget:
                break

        return current_solution, current_fitness, evals

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
                    candidate, candidate_fitness, sa_evals = self.simulated_annealing(func, population[i])
                    evals += sa_evals
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
                scale_factor = 0.5 + 0.3 * np.random.rand()
                crossover_rate = 0.8 + 0.1 * np.random.rand()
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
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