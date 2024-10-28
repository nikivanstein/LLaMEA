import numpy as np

class AdaptiveChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.F_initial = 0.7  # Adaptive scaling factor
        self.CR_initial = 0.8  # Adaptive crossover probability
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

                # Adaptive Parameters
                F = self.F_initial + 0.2 * np.sin(np.pi * evaluations / self.budget)
                CR = self.CR_initial + 0.2 * np.cos(np.pi * evaluations / self.budget)

                # Chaotic Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                chaotic_factor = np.sin(evaluations / self.budget * np.pi)
                mutant = np.clip(x0 + F * (x1 - x2) + chaotic_factor * (global_best - x0), self.lower_bound, self.upper_bound)

                # Adaptive Crossover
                crossover_mask = np.random.rand(self.dim) < (CR + 0.05 * (1 - evaluations / self.budget))
                trial = np.where(crossover_mask, mutant, population[i])

                # Chaotic Refinement
                if np.random.rand() < 0.4:
                    trial = self.chaotic_refinement(trial, func, evaluations / self.budget)

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

    def chaotic_refinement(self, solution, func, progress):
        step_size = 0.08 * (self.upper_bound - self.lower_bound) * np.abs(np.sin(progress * np.pi))
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(5):
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness

        return best_solution