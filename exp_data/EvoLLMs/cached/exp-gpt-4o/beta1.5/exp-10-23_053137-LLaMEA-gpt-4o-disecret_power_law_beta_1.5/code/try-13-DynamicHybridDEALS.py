import numpy as np

class DynamicHybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        adaptation_interval = self.budget // 10  # Update every 10% of budget
        
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
                if np.random.rand() < 0.2:  # 20% chance to refine the trial solution
                    trial = self.local_search(trial, func)

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Dynamic Parameter Adaptation
            if evaluations % adaptation_interval == 0:
                self.adapt_parameters(fitness)

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

    def adapt_parameters(self, fitness):
        # Adaptive tuning of F and CR based on population diversity
        diversity = np.std(fitness)
        self.F = 0.5 + 0.1 * (diversity / max(1, np.mean(fitness)))
        self.CR = 0.7 + 0.2 * (diversity / max(1, np.mean(fitness)))
        # Ensure parameters remain in valid range
        self.F = np.clip(self.F, 0.4, 1.0)
        self.CR = np.clip(self.CR, 0.5, 1.0)