import numpy as np
from scipy.optimize import minimize

class EnhancedPSOHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.3  # Switch to PSO after 30% of budget

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

        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()

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
                # Particle Swarm Optimization
                w, c1, c2 = 0.5, 1.5, 1.5
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = w * velocities[i] + c1 * r1 * (personal_best[i] - population[i]) + c2 * r2 * (best_solution - population[i])
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                    
                    fitness[i] = func(population[i])
                    evals += 1
                    if fitness[i] < personal_best_fitness[i]:
                        personal_best[i] = population[i].copy()
                        personal_best_fitness[i] = fitness[i]
                        if fitness[i] < best_fitness:
                            best_fitness = fitness[i]
                            best_solution = personal_best[i].copy()
                    if evals >= self.budget:
                        break

        return best_solution