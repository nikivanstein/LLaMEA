import numpy as np

class ECMAES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lam = 20
        self.sigma = 0.3
        self.sigma_init = 0.3
        self.c_c = 0.4
        self.c_s = 0.3
        self.c_1 = 2.0
        self.c_mu = 2.0
        self.weights = np.log(self.lam + 1) - np.log(np.arange(1, self.lam + 1))
        self.weights /= np.sum(self.weights)

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        x_mean = np.random.uniform(lower_bound, upper_bound, self.dim)
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        D = np.diag(np.ones(self.dim) * self.sigma)

        evals = 0
        while evals < self.budget:
            offspring = np.random.multivariate_normal(np.zeros(self.dim), C, self.lam)
            offspring = x_mean + self.sigma * offspring.T
            offspring = np.clip(offspring, lower_bound, upper_bound)
            f_vals = np.array([func(off) for off in offspring.T])
            evals += self.lam

            sorted_indices = np.argsort(f_vals)
            x_k = np.mean(offspring[:, sorted_indices[:self.lam // 2]], axis=1)
            y_k = np.mean(offspring[:, sorted_indices[:self.lam // 2]] - x_mean.reshape(-1, 1), axis=1)

            p_sigma = (1 - self.c_s) * p_sigma + np.sqrt(self.c_s * (2 - self.c_s) * self.lam) * np.dot(C, y_k)
            C = (1 - self.c_c - self.c_1) * C + self.c_c * np.outer(p_sigma, p_sigma) + self.c_1 * np.sum(
                self.weights[i] * np.outer(offspring[:, i] - x_k, offspring[:, i] - x_k) for i in range(self.lam))

            D = np.diag(np.sqrt(np.diag(C)))
            x_mean = x_k.copy()
            self.sigma = self.sigma * np.exp((self.c_mu / np.sqrt(self.dim)) * (np.linalg.norm(p_sigma) / np.sqrt(self.lam) - 1))

        return x_mean