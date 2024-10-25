import numpy as np

class EnhancedHybridDEASAV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(0.6 * dim)  # Slightly larger population size
        self.prob_crossover = 0.85  # Slightly reduced crossover probability
        self.F = 0.9  # Increased differential weight for more aggressive mutation
        self.current_evaluations = 0
        self.temperature = 1.0
        self.diversity_factor = 0.15  # Further increased diversity for enhanced exploration

    def __call__(self, func):
        # Initialize population with evaluation
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                # Select indices for mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Introduce dynamic neighborhood exploration
                if np.random.rand() < self.diversity_factor:
                    local_mask = np.random.rand(self.dim) < 0.5
                    mutant[local_mask] = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)[local_mask]
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Perform crossover
                crossover_mask = np.random.rand(self.dim) < self.prob_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial solution and calculate acceptance
                trial_fitness = func(trial)
                self.current_evaluations += 1
                delta_e = trial_fitness - fitness[i]
                # Adaptive simulated annealing acceptance
                acceptance_probability = np.exp(-delta_e / (self.temperature + 1e-9))

                # Adaptive temperature schedule
                self.temperature *= 0.95  # More aggressive cooling rate

                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution