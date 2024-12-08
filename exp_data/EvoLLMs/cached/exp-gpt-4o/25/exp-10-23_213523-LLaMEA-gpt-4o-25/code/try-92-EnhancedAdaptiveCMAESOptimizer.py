import numpy as np

class EnhancedAdaptiveCMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 4 + int(3 * np.log(self.dim))
        self.covariance_matrix = np.eye(self.dim)
        self.adaptive_step_size = 0.5
        self.mean = np.random.uniform(-5.0, 5.0, self.dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0
        self.fitness = np.full(self.population_size, np.inf)
        self.elite_count = max(1, int(self.population_size * 0.1))

    def __call__(self, func):
        weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.5)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        p_c = np.zeros(self.dim)
        p_sigma = np.zeros(self.dim)

        while self.eval_count < self.budget:
            samples = np.random.multivariate_normal(self.mean, self.adaptive_step_size**2 * self.covariance_matrix, self.population_size)
            mirrored_samples = 2 * self.mean - samples  # Stochastic mirrored sampling
            samples = np.concatenate((samples, mirrored_samples), axis=0)
            samples = np.clip(samples, self.lower_bound, self.upper_bound)

            for i in range(len(samples)):
                if self.eval_count >= self.budget:
                    break
                self.fitness[i] = func(samples[i])
                self.eval_count += 1

            indices = np.argsort(self.fitness)
            selected = samples[indices[:len(weights)]]
            self.mean = np.dot(weights, selected)

            y_k = (selected - self.mean) / self.adaptive_step_size
            c_y = np.sum(weights[:, None, None] * (y_k[:, :, None] @ y_k[:, None, :]), axis=0)
            self.covariance_matrix = (1 - c_1 - c_mu) * self.covariance_matrix + c_1 * (p_c[:, None] @ p_c[None, :]) + c_mu * c_y

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(np.linalg.cholesky(self.covariance_matrix), self.mean - selected[0]) / self.adaptive_step_size
            self.adaptive_step_size *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))

            if self.eval_count < self.budget:
                elite_samples = samples[indices[:self.elite_count]]
                for elite_sample in elite_samples:
                    self.fitness = np.append(self.fitness, func(elite_sample))
                    self.eval_count += 1
                    if self.eval_count >= self.budget:
                        break

        best_idx = np.argmin(self.fitness)
        return samples[best_idx], self.fitness[best_idx]