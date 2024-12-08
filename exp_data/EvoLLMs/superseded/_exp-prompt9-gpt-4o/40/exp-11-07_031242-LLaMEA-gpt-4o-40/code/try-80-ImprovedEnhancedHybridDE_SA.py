import numpy as np

class ImprovedEnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30  # Increased initial population for better exploration
        self.min_population_size = 10  # Slightly increased minimum population for robustness
        self.de_mutation_factor = 0.75  # Enhanced DE mutation factor for larger steps
        self.cr = 0.85  # Slightly reduced crossover rate for improved fine-tuning
        self.initial_temperature = 80.0  # Adjusted starting temperature for balanced exploration
        self.temperature_decay = 0.95  # Slower decay to maintain exploration longer

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        # Utilizing a logistic map for chaotic initialization
        chaotic_sequence = np.mod(0.4 * np.arange(population_size * self.dim), 1.0)
        population = chaotic_sequence.reshape(population_size, self.dim) * (self.upper_bound - self.lower_bound) + self.lower_bound
        fitness = np.array([func(ind) for ind in population])
        evals_used = population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx, temperature):
            indices = list(range(population_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            scale = (1.0 + np.exp(-fitness[target_idx])) / (1.0 + np.exp(-fitness))
            mutant = np.clip(a + scale[target_idx] * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
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

            # Gradual convergence with adaptive population control
            population_size = max(self.min_population_size, int(self.initial_population_size * (1 - evals_used / self.budget)))
            population = population[:population_size]
            fitness = fitness[:population_size]

        return best_solution, best_fitness