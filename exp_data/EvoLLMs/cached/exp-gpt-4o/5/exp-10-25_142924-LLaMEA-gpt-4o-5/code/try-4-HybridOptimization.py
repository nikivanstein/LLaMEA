import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_de = int(budget * 0.7)
        self.num_nm = budget - self.num_de

    def chaotic_map(self, iteration):
        return 0.7 * (1 - (iteration / self.num_de)) + 0.3 * np.sin(iteration)

    def __call__(self, func):
        # Adaptive Differential Evolution (DE) parameters
        population_size = 10 * self.dim
        F = 0.5  # Initial Differential weight
        CR = 0.7  # Initial Crossover probability
        delta_F = 0.05  # Adaptation step for F
        delta_CR = 0.03  # Adaptation step for CR

        # Initialize DE population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # DE loop with dynamic mutation strategy
        iteration = 0
        while evaluations < self.num_de:
            for i in range(population_size):
                idxs = np.random.choice(list(set(range(population_size)) - {i}), 3, replace=False)
                a, b, c = population[idxs]
                F = self.chaotic_map(iteration)  # Use chaotic map for F
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    CR = min(1.0, CR + delta_CR)  # Increase CR for successful trials
                else:
                    CR = max(0.1, CR - delta_CR)  # Decrease CR for unsuccessful trials

                if evaluations >= self.num_de:
                    break
            iteration += 1

        # Take the best solution found by DE
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        # Sequential Quadratic Programming for exploitation
        if evaluations < self.budget:
            result = minimize(func, best_solution, method='SLSQP', bounds=[(self.lower_bound, self.upper_bound)]*self.dim, options={'maxiter': self.num_nm})
            evaluations += result.nfev
            if result.success and result.fun < fitness[best_idx]:
                best_solution = result.x

        return best_solution