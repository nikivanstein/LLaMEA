import numpy as np

class OptimizedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 35  # Increased population size for diversity
        self.min_population_size = 5       # Smaller minimum population size
        self.de_mutation_factor = 0.9      # Adjusted DE mutation factor
        self.cr = 0.9                      # Slightly higher crossover rate
        self.initial_temperature = 100.0   # Higher initial temperature for more exploration
        self.temperature_decay = 0.93      # Faster decay of temperature

    def __call__(self, func):
        np.random.seed(42)  # Changed seed for different randomness
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx, temperature):
            indices = list(range(population_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.de_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[target_idx])
            return trial

        temperature = self.initial_temperature
        while evals_used < self.budget:
            for i in range(population_size):
                trial = de_mutation_and_crossover(i, temperature)
                trial_fitness = func(trial)
                evals_used += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                temperature *= self.temperature_decay

                if evals_used >= self.budget:
                    break

            # Adapt population size dynamically for more aggressive convergence
            population_size = max(self.min_population_size, int(self.initial_population_size * (1 - evals_used / self.budget) ** 1.5))
            population = population[:population_size]
            fitness = fitness[:population_size]

        return best_solution, best_fitness