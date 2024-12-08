import numpy as np

class HybridDEALS:
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

                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Local Search
                if np.random.rand() < 0.15:  # Adjusted to 15% chance to refine the trial solution
                    trial = self.local_search(trial, func)

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive parameter adjustment
                if evaluations % (self.population_size * 5) == 0:
                    self.F = np.random.uniform(0.4, 0.9)
                    self.CR = np.random.uniform(0.8, 1.0)

        return population[np.argmin(fitness)]

    def local_search(self, solution, func):
        # Adaptive local search: small perturbations in random directions
        step_size = 0.05 * (self.upper_bound - self.lower_bound)  # Adjusted step size for local search
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(10):  # Perform more local steps for thorough refinement
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness
        
        return best_solution