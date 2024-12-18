import numpy as np

class AdaptiveDE_CMA_V3_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4 + int(3 * np.log(self.dim)), 10)
        self.mutation_factor = 0.7
        self.crossover_rate = 0.9
        self.sigma_initial = 0.2
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.momentum = np.zeros((self.population_size, self.dim))

    def adaptive_mutation_factor(self, evals):
        return 0.7 + 0.2 * np.sin(np.pi * evals / self.budget)

    def adaptive_crossover_rate(self, evals):
        return 0.9 - 0.4 * (evals / self.budget)

    def __call__(self, func):
        evals = 0
        learning_rate = 0.01
        diversity_threshold = 0.1
        while evals < self.budget:
            if evals + self.population_size > self.budget:
                break

            fitness = np.apply_along_axis(func, 1, self.population)
            evals += self.population_size

            if np.min(fitness) < self.best_fitness:
                self.best_fitness = np.min(fitness)
                self.best_solution = self.population[np.argmin(fitness)].copy()

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                noise = np.random.normal(0, 0.01, self.dim)
                mutation_factor = self.adaptive_mutation_factor(evals)
                mutant_vector = np.clip(a + mutation_factor * (b - c) + noise, self.lower_bound, self.upper_bound)
                crossover_vector = np.where(np.random.rand(self.dim) < self.adaptive_crossover_rate(evals), mutant_vector, self.population[i])
                
                if np.random.rand() < 0.1:
                    crossover_vector = mutant_vector
                
                if func(crossover_vector) < fitness[i]:
                    self.population[i] = crossover_vector
                    evals += 1
                    if evals >= self.budget:
                        break

            learning_rate = 0.007 + 0.01 * (np.exp(-evals / self.budget))

            for i in range(self.population_size):
                diversity = np.linalg.norm(self.best_solution - self.population[i])
                if diversity > diversity_threshold:
                    self.momentum[i] = 0.9 * self.momentum[i] + learning_rate * (self.best_solution - self.population[i])
                self.population[i] += self.momentum[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            cov_matrix = np.cov(self.population.T)
            mean_vector = np.mean(self.population, axis=0)
            adaptive_sigma = self.sigma_initial**2 + evals / self.budget + 0.01
            self.population = np.random.multivariate_normal(mean_vector, adaptive_sigma * cov_matrix, self.population_size)
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

        return self.best_solution