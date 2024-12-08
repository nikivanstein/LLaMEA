import numpy as np

class EnhancedDEALS:
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
                mutant = np.clip(x0 + self.adaptive_F(evaluations) * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.adaptive_CR(evaluations)
                trial = np.where(crossover_mask, mutant, population[i])

                # Local Search
                if np.random.rand() < 0.2:  # 20% chance to refine the trial solution
                    trial = self.local_search(trial, func)

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)]

    def local_search(self, solution, func):
        # Adaptive local search: small perturbations in random directions
        step_size = 0.1 * (self.upper_bound - self.lower_bound)
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(5):  # Perform a small number of local steps
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness
        
        return best_solution

    def adaptive_F(self, evaluations):
        # Adapt F dynamically based on the current evaluation budget
        return 0.5 + 0.5 * (evaluations / self.budget)

    def adaptive_CR(self, evaluations):
        # Adapt CR dynamically based on the current evaluation budget
        return 0.9 - 0.8 * (evaluations / self.budget)