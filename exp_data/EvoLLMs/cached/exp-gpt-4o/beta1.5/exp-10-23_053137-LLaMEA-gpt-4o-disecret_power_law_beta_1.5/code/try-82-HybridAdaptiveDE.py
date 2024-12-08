import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.F = 0.7  # Scaling factor for exploration
        self.CR = 0.9  # Crossover probability for exploitation
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

                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover with Dynamic Strategy
                crossover_mask = np.random.rand(self.dim) < (self.CR + 0.05 * (1 - evaluations / self.budget))
                trial = np.where(crossover_mask, mutant, population[i])

                # Stochastic Regional Refinement
                if np.random.rand() < 0.3:  # Adjusted local search probability
                    trial = self.stochastic_regional_refinement(trial, func, evaluations / self.budget)

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

    def stochastic_regional_refinement(self, solution, func, progress):
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * (1 - progress ** 1.2)  # Refinement step
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(2):  # Iterations for efficiency
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution