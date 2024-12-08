import numpy as np
from scipy.optimize import minimize
from cma import CMAEvolutionStrategy

class ImprovedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.3  # Earlier switch to Differential Evolution
        self.cma_init_sigma = 0.5   # Initial sigma for CMA-ES

    def __call__(self, func):
        # Initialize CMA-ES
        es = CMAEvolutionStrategy(self.dim * [0], self.cma_init_sigma)
        evals = 0
        best_solution = None
        best_fitness = float('inf')

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # CMA-ES phase
                if not es.stop():
                    solutions = es.ask()
                    solutions = np.clip(solutions, self.lower_bound, self.upper_bound)
                    fitness_values = [func(s) for s in solutions]
                    evals += len(solutions)
                    es.tell(solutions, fitness_values)
                    es.logger.add()
                    es.disp()
                    best_idx = np.argmin(fitness_values)
                    if fitness_values[best_idx] < best_fitness:
                        best_fitness = fitness_values[best_idx]
                        best_solution = solutions[best_idx].copy()
            else:
                # Modified Differential Evolution with adaptive mutation
                if best_solution is None:
                    population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                    fitness = np.apply_along_axis(func, 1, population)
                    evals += self.population_size
                    best_idx = np.argmin(fitness)
                    best_solution = population[best_idx].copy()
                    best_fitness = fitness[best_idx]

                scale_factor = 0.6 + 0.2 * np.random.rand()
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])
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