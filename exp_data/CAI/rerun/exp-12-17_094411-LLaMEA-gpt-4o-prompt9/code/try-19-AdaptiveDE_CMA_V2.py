import numpy as np

class AdaptiveDE_CMA_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4 + int(3 * np.log(self.dim)), 10)
        self.mutation_factor = 0.7  # Updated initial mutation factor
        self.crossover_rate = 0.9  # Updated crossover probability
        self.sigma_initial = 0.2  # Adjusted initial sigma
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.momentum = np.zeros((self.population_size, self.dim))  # Momentum for exploration

    def __call__(self, func):
        evals = 0
        learning_rate = 0.01  # Updated adaptive learning rate
        diversity_threshold = 0.1  # Introduced diversity threshold
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
                noise = np.random.normal(0, 0.01, self.dim)  # Noise infusion into mutation
                mutant_vector = np.clip(a + self.mutation_factor * (b - c) + noise, self.lower_bound, self.upper_bound)
                crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                if func(crossover_vector) < fitness[i]:
                    self.population[i] = crossover_vector
                    evals += 1
                    if evals >= self.budget:
                        break

            self.mutation_factor = 0.5 + 0.3 * np.exp(-evals / self.budget)  # Adjusted annealing adaptation
            learning_rate = 0.007 + 0.01 * (np.exp(-evals / self.budget))  # Refined learning rate

            # Momentum-based exploration adjustment
            for i in range(self.population_size):
                diversity = np.linalg.norm(self.best_solution - self.population[i])
                if diversity > diversity_threshold:
                    self.momentum[i] = 0.9 * self.momentum[i] + learning_rate * (self.best_solution - self.population[i])
                self.population[i] += self.momentum[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            cov_matrix = np.cov(self.population.T)
            mean_vector = np.mean(self.population, axis=0)
            self.population = np.random.multivariate_normal(mean_vector, (self.sigma_initial**2 + evals / self.budget) * cov_matrix, self.population_size)
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

        return self.best_solution