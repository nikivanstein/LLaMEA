import numpy as np

class AdaptiveDE_CMA_V2_Enhanced_Modified_V7:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4 + int(3 * np.log(self.dim)), 12)
        self.mutation_factor = 0.75
        self.crossover_rate = 0.85
        self.sigma_initial = 0.2
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.momentum = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        evals = 0
        learning_rate = 0.015
        diversity_threshold = 0.12
        decay_rate = 0.99

        while evals < self.budget:
            if evals + self.population_size > self.budget:
                break

            fitness = np.apply_along_axis(func, 1, self.population)
            evals += self.population_size

            if np.min(fitness) < self.best_fitness:
                self.best_fitness = np.min(fitness)
                self.best_solution = self.population[np.argmin(fitness)].copy()

            convergence_progress = evals / self.budget
            self.mutation_factor *= decay_rate

            avg_diversity = np.mean([np.linalg.norm(self.population[i] - self.best_solution) for i in range(self.population_size)])
            noise_scale_factor = 0.45 * (1 - convergence_progress) + 0.1 * avg_diversity / self.dim

            # Population size adaptation added
            self.population_size = max(4 + int(3 * np.log(self.dim)) - int(convergence_progress * 2), 12)

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                noise = np.random.normal(0, noise_scale_factor * np.sqrt(self.dim), self.dim)
                mutation_intensity = 1 + (evals / self.budget)
                mutant_vector = np.clip(a + self.mutation_factor * (b - c) * mutation_intensity + noise, self.lower_bound, self.upper_bound)
                
                variance_factor = np.var(self.population) / (self.upper_bound - self.lower_bound)
                adaptive_crossover_rate = self.crossover_rate + 0.1 * variance_factor

                crossover_vector = np.where(np.random.rand(self.dim) < adaptive_crossover_rate, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.12:
                    crossover_vector = mutant_vector
                
                if func(crossover_vector) < fitness[i]:
                    self.population[i] = crossover_vector
                    evals += 1
                    if evals >= self.budget:
                        break

            learning_rate = 0.01 + 0.012 * (np.exp(-evals / self.budget))

            for i in range(self.population_size):
                diversity = np.linalg.norm(self.best_solution - self.population[i])
                if diversity > diversity_threshold:
                    self.momentum[i] = 0.85 * self.momentum[i] + learning_rate * (self.best_solution - self.population[i])
                self.population[i] += self.momentum[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            cov_matrix = np.cov(self.population.T)
            mean_vector = np.mean(self.population, axis=0)
            adaptive_sigma = self.sigma_initial**2 + evals / self.budget + 0.015
            self.population = np.random.multivariate_normal(mean_vector, adaptive_sigma * cov_matrix, self.population_size)
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

        return self.best_solution