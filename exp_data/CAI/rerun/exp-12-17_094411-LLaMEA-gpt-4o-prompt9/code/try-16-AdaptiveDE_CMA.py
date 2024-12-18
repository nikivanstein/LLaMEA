import numpy as np

class AdaptiveDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 4 + int(3 * np.log(self.dim)) + int(self.budget * 0.002)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.sigma_initial = 0.3
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.diversity_threshold = 1e-6  # New threshold for diversity control

    def __call__(self, func):
        evals = 0
        stagnation_counter = 0  # Counter for stagnation detection

        while evals < self.budget:
            if evals + self.population_size > self.budget:
                break

            fitness = np.apply_along_axis(func, 1, self.population)
            evals += self.population_size

            if np.min(fitness) < self.best_fitness:
                self.best_fitness = np.min(fitness)
                self.best_solution = self.population[np.argmin(fitness)].copy()
                stagnation_counter = 0  # Reset counter on improvement
            else:
                stagnation_counter += 1  # Increment counter on stagnation

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                if func(crossover_vector) < fitness[i]:
                    self.population[i] = crossover_vector
                    evals += 1
                    if evals >= self.budget:
                        break

            self.mutation_factor = 0.5 + 0.3 * np.exp(-evals / self.budget)

            cov_matrix = np.cov(self.population.T)
            mean_vector = np.mean(self.population, axis=0)
            self.population = np.random.multivariate_normal(mean_vector, self.sigma_initial**2 * cov_matrix, self.population_size)
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            # Adaptive population control based on diversity
            population_std = np.std(self.population, axis=0)
            if np.all(population_std < self.diversity_threshold) or stagnation_counter > 10:
                self.population_size = min(int(self.population_size * 1.1), int(self.budget / (evals + 1)))
                self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                stagnation_counter = 0  # Reset stagnation counter

        return self.best_solution