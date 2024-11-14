import numpy as np
from scipy.stats import norm

class ABODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.eval_count = 0
        self.surrogate_samples = 50
        self.noise_var = 1e-6

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def mutate(self, individual, population):
        indices = np.random.choice(self.population_size, 3, replace=False)
        x1, x2, x3 = population[indices]
        mutant = x1 + self.mutation_factor * (x2 - x3)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def bayesian_surrogate(self, X, y, X_new):
        mu = np.mean(y)
        K = np.exp(-np.sum((X[:, None] - X_new[None, :])**2, axis=2) / (2 * self.noise_var))
        K_inv = np.linalg.inv(K + self.noise_var * np.eye(len(X)))
        mu_pred = mu + K_inv @ (y - mu)
        std_pred = np.sqrt(np.abs(1 - np.diag(K @ K_inv)))
        return mu_pred, std_pred

    def expected_improvement(self, mu, std, y_best):
        with np.errstate(divide='warn'):
            imp = mu - y_best
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
        return ei

    def optimize_surrogate(self, X, y, func):
        X_samples = np.random.uniform(self.lower_bound, self.upper_bound, (self.surrogate_samples, self.dim))
        mu_pred, std_pred = self.bayesian_surrogate(X, y, X_samples)
        y_best = np.min(y)
        ei = self.expected_improvement(mu_pred, std_pred, y_best)
        best_idx = np.argmax(ei)
        return X_samples[best_idx]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best = population[best_idx]

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                mutant = self.mutate(population[i], population)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if self.eval_count < self.budget:
                surrogate_opt = self.optimize_surrogate(population, fitness, func)
                surrogate_fitness = func(surrogate_opt)
                self.eval_count += 1
                if surrogate_fitness < fitness[best_idx]:
                    population[best_idx] = surrogate_opt
                    fitness[best_idx] = surrogate_fitness

        return population[np.argmin(fitness)]