import numpy as np

class EnhancedHybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation with dynamic feedback control
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                progress = evaluations / self.budget
                adaptive_F = self.F * (1 - progress) + 0.1 * progress
                mutant = np.clip(x0 + adaptive_F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Adaptive Stochastic Local Search with increased adaptive perturbation
                if np.random.rand() < 0.25:  # 25% chance to refine the trial solution
                    trial = self.adaptive_stochastic_local_search(trial, func, progress)

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)]

    def adaptive_stochastic_local_search(self, solution, func, progress):
        # Adaptive Stochastic Local Search: dynamically adjusts perturbation size
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * (2 - progress ** 2)
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(4):  # Slightly increase local steps to enhance search precision
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness
        
        return best_solution