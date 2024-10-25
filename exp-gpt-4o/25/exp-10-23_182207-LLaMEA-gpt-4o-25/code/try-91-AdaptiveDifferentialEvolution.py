import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 2)
        self.mutation_factor = 0.9
        self.crossover_rate = 0.8
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.15
        self.noise_factor = 0.1
        self.min_population_size = self.population_size // 2
        self.max_population_size = self.population_size * 2

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            fitness_values = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                noise = np.random.normal(0, self.noise_factor, self.dim)
                trial_noisy = trial + noise
                trial_noisy = np.clip(trial_noisy, *self.bounds)
                
                trial_fitness = func(trial_noisy)
                evaluations += 1

                if trial_fitness < fitness_values[i]:
                    new_population[i] = trial_noisy
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_noisy

            if stagnation_counter > self.population_size // 2:
                self.population_size = min(self.population_size + 1, self.max_population_size)
                self.adaptive_sigma = min(self.adaptive_sigma * 1.3, 1.0)
                self.noise_factor = min(self.noise_factor * 1.2, 0.2)
            else:
                self.population_size = max(self.population_size - 1, self.min_population_size)
                self.adaptive_sigma = max(self.adaptive_sigma * 0.7, 0.01)
                self.noise_factor = max(self.noise_factor * 0.9, 0.05)

            self.population = new_population

        return self.best_solution, self.best_fitness