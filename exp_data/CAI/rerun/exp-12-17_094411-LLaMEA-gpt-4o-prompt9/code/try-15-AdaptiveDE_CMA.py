import numpy as np

class AdaptiveDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        # Dynamic population size based on budget and dimension influences exploration and exploitation
        self.population_size = 4 + int(3 * np.log(self.dim)) + int(self.budget * 0.002)
        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_rate = 0.9  # Crossover probability
        self.sigma_initial = 0.3  # Initial step size for CMA-ES
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            if evals + self.population_size > self.budget:
                break

            # Evaluate current population
            fitness = np.apply_along_axis(func, 1, self.population)
            evals += self.population_size

            # Update the best solution
            if np.min(fitness) < self.best_fitness:
                self.best_fitness = np.min(fitness)
                self.best_solution = self.population[np.argmin(fitness)].copy()

            # Differential Evolution Mutation and Crossover
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

            # Annealing-inspired adaptation of DE parameters
            self.mutation_factor = 0.5 + 0.3 * np.exp(-evals / self.budget)  # Annealing-style adaptation

            # Covariance Matrix Adaptation (CMA) inspired update
            cov_matrix = np.cov(self.population.T)
            mean_vector = np.mean(self.population, axis=0)
            self.population = np.random.multivariate_normal(mean_vector, self.sigma_initial**2 * cov_matrix, self.population_size)
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

        return self.best_solution