import numpy as np

class HybridDE_ACME:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf

    def differential_evolution_step(self, func):
        # Introduce adaptive tuning for mutation factor and crossover probability
        current_mutation_factor = self.mutation_factor * (1 + np.random.rand() * 0.1)
        current_crossover_probability = self.crossover_probability * (1 - self.budget / (self.budget + 1)) * (1 + np.random.rand() * 0.1)
        volatility_factor = np.std(self.population, axis=0).mean() / 5.0
        for i in range(self.population_size):
            if self.budget <= 0:
                break
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = np.clip(a + current_mutation_factor * volatility_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < current_crossover_probability
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            self.budget -= 1
            if trial_fitness < self.fitness[i]:
                self.fitness[i] = trial_fitness
                self.population[i] = trial
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

    def adaptive_cma_step(self, func):
        if self.best_solution is None:
            self.best_solution = np.mean(self.population, axis=0)
        covariance_matrix = np.cov(self.population.T)
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        step_size = np.sqrt(np.max(eigvals))
        while self.budget > 0:
            samples = np.random.multivariate_normal(self.best_solution, step_size * covariance_matrix, self.population_size)
            samples = np.clip(samples, self.lower_bound, self.upper_bound)
            for sample in samples:
                if self.budget <= 0:
                    break
                sample_fitness = func(sample)
                self.budget -= 1
                if sample_fitness < self.best_fitness:
                    self.best_fitness = sample_fitness
                    self.best_solution = sample

    def __call__(self, func):
        while self.budget > 0:
            self.differential_evolution_step(func)
            self.adaptive_cma_step(func)
        return self.best_solution