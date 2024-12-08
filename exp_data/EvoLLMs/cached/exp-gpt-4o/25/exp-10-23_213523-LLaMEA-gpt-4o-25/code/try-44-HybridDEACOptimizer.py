import numpy as np

class HybridDEACOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.log(self.dim))
        self.covariance_matrix = np.eye(self.dim)
        self.step_size = 0.6
        self.mean = np.random.uniform(-5.0, 5.0, self.dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0
        self.fitness = np.full(self.population_size, np.inf)

    def __call__(self, func):
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 1) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_1 = 2 / ((self.dim + 1.5)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        p_sigma = np.zeros(self.dim)
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for i in range(self.population_size):
            self.fitness[i] = func(population[i])
            self.eval_count += 1

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    population[i] = trial

            indices = np.argsort(self.fitness)
            selected = population[indices[:len(weights)]]
            self.mean += np.dot(weights, selected - self.mean)

            y_k = (selected - self.mean) / self.step_size
            c_y = np.sum(weights[:, None, None] * (y_k[:, :, None] @ y_k[:, None, :]), axis=0)
            self.covariance_matrix = (1 - c_1 - c_mu) * self.covariance_matrix + c_1 * np.outer(self.mean - selected[0], self.mean - selected[0]) + c_mu * c_y

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(np.linalg.cholesky(self.covariance_matrix), self.mean - selected[0]) / self.step_size
            self.step_size *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))

        best_idx = np.argmin(self.fitness)
        return population[best_idx], self.fitness[best_idx]