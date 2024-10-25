import numpy as np

class DualPhaseAGIDCMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 4 + int(3 * np.log(self.dim))
        self.covariance_matrix = np.eye(self.dim)
        self.step_size = 0.5
        self.mean = np.random.uniform(-5.0, 5.0, self.dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0
        self.fitness = np.full(self.population_size, np.inf)
        self.crossover_rate = 0.6  # Adjust crossover rate
        self.mutation_prob = 0.1  # Introduce mutation probability

    def __call__(self, func):
        weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 1) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.5)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        p_c = np.zeros(self.dim)
        p_sigma = np.zeros(self.dim)

        while self.eval_count < self.budget:
            samples = np.random.multivariate_normal(self.mean, self.step_size**2 * self.covariance_matrix, self.population_size)
            samples = np.clip(samples, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                # Introduce a dual-phase strategy: crossover and mutation
                if np.random.rand() < self.crossover_rate:
                    crossover_idx = np.random.randint(0, self.population_size)
                    samples[i] = 0.5 * (samples[i] + samples[crossover_idx])
                if np.random.rand() < self.mutation_prob:
                    samples[i] += np.random.normal(0, self.step_size, self.dim)

                self.fitness[i] = func(samples[i])
                self.eval_count += 1

            indices = np.argsort(self.fitness)
            selected = samples[indices[:len(weights)]]
            self.mean += np.dot(weights, selected - self.mean)

            y_k = (selected - self.mean) / self.step_size
            c_y = np.sum(weights[:, None, None] * (y_k[:, :, None] @ y_k[:, None, :]), axis=0)
            self.covariance_matrix = (1 - c_1 - c_mu) * self.covariance_matrix + c_1 * (p_c[:, None] @ p_c[None, :]) + c_mu * c_y

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(np.linalg.cholesky(self.covariance_matrix), self.mean - selected[0]) / self.step_size
            self.step_size *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))

        best_idx = np.argmin(self.fitness)
        return samples[best_idx], self.fitness[best_idx]