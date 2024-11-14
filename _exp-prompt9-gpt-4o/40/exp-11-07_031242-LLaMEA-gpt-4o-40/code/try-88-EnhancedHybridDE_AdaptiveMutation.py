import numpy as np

class EnhancedHybridDE_AdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30  # Increased initial population for better diversity
        self.min_population_size = 10  # Higher minimum population size for robustness
        self.initial_de_mutation_factor = 0.7  # Adaptive mutation factor starting point
        self.cr = 0.85  # Adjusted crossover rate for enhanced exploration
        self.initial_temperature = 120.0  # Higher initial temperature for broader exploration
        self.temperature_decay = 0.95  # Slower decay for sustained exploration

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx, de_mutation_factor):
            indices = list(range(population_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + de_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[target_idx])
            return trial

        temperature = self.initial_temperature
        while evals_used < self.budget:
            de_mutation_factor = self.initial_de_mutation_factor * (1 - evals_used / self.budget) + 0.4  # Adaptive factor
            for i in range(population_size):
                trial = de_mutation_and_crossover(i, de_mutation_factor)
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

            population_size = max(self.min_population_size, int(self.initial_population_size * (1 - 0.5 * (evals_used / self.budget))))
            population = population[:population_size]
            fitness = fitness[:population_size]

        return best_solution, best_fitness