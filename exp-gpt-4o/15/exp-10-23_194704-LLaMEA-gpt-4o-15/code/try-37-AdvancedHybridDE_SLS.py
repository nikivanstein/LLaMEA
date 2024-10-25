import numpy as np

class AdvancedHybridDE_SLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.de_cross_over_rate = 0.85
        self.de_f = 0.4 + 0.2 * np.random.rand()  # Increased stochastic factor adjustment
        self.simplex_size = dim + 2  # Adjusted simplex size for diversity

    def __call__(self, func):
        budget_used = 0

        # Initialize population with varied range
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        budget_used += self.population_size

        while budget_used < self.budget:
            # Sort population by fitness
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]

            # Adaptive Differential Evolution step
            for i in range(self.population_size):
                if budget_used >= self.budget:
                    break
                
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.de_f * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.de_cross_over_rate
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if budget_used >= self.budget:
                break

            # Refined Stochastic Local Search
            for i in range(min(6, self.population_size)):  # Apply to top solutions
                if budget_used >= self.budget:
                    break
                local_perturbation = np.random.normal(0, 0.12, self.dim)
                candidate = np.clip(population[i] + local_perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                budget_used += 1
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]