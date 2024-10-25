import numpy as np

class StochasticMultiPhaseOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 + 5 * self.dim  # Slightly increased population size for diversity
        self.F = 0.6  # Adjusted differential weight for better mutation
        self.CR = 0.85  # Slightly reduced crossover rate for exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        global_best = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation with Swarm Influence
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2) + 0.15 * (global_best - x0), self.lower_bound, self.upper_bound)  # Increased swarm influence

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Adaptive Stochastic Local Search
                if np.random.rand() < 0.4:  # Further increased chance to refine the trial solution
                    trial = self.adaptive_stochastic_local_search(trial, func, evaluations / self.budget)

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness

        return global_best

    def adaptive_stochastic_local_search(self, solution, func, progress):
        step_size = 0.08 * (self.upper_bound - self.lower_bound) * (1 - progress ** 2)  # Adjusted step size
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(4):  # Increased number of local search iterations
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution