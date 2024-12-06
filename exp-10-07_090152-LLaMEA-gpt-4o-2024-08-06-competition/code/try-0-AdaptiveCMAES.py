import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, population_size=None):
        self.budget = budget
        self.dim = dim
        self.lambda_ = population_size if population_size else 4 + int(3 * np.log(dim))
        self.sigma = 0.5
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.lambda_)
        self.cmu = min(1 - self.c1, 2 * (self.lambda_ - 2 + 1 / self.lambda_) / ((dim + 2) ** 2 + self.lambda_))
        self.cc = (4 + self.lambda_ / dim) / (dim + 4 + 2 * self.lambda_ / dim)
        self.csigma = (self.lambda_ + 2) / (dim + self.lambda_ + 5)
        self.dsigma = 1 + 2 * max(0, np.sqrt((self.lambda_ - 1) / (dim - 1)) - 1) + self.csigma
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        self.weights = np.log(self.lambda_ / 2 + 1) - np.log(np.arange(1, self.lambda_ + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.pc = np.zeros(dim)
        self.psigma = np.zeros(dim)
        self.C = np.eye(dim)
        self.B, self.D = np.linalg.eigh(self.C)
        self.BD = self.B * self.D
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        mean = np.random.uniform(-100, 100, self.dim)
        for _ in range(self.budget // self.lambda_):
            arz = np.random.randn(self.lambda_, self.dim)
            arx = mean + self.sigma * np.dot(arz, self.BD)
            arx = np.clip(arx, -100, 100)
            fitness = np.array([func(x) for x in arx])
            order = np.argsort(fitness)
            if fitness[order[0]] < self.best_value:
                self.best_value = fitness[order[0]]
                self.best_solution = arx[order[0]]
            mean = np.dot(self.weights, arx[order[:len(self.weights)]])
            zmean = np.dot(self.weights, arz[order[:len(self.weights)]])
            self.psigma = (1 - self.csigma) * self.psigma + np.sqrt(self.csigma * (2 - self.csigma) * self.mu_eff) * np.dot(self.B, zmean)
            self.sigma *= np.exp((np.linalg.norm(self.psigma) / self.chiN - 1) * self.csigma / self.dsigma)
            hsig = int((np.linalg.norm(self.psigma) / np.sqrt(1 - (1 - self.csigma) ** (2 * (_ + 1))) / self.chiN) < (1.4 + 2 / (self.dim + 1)))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * np.dot(self.B, zmean)
            artmp = (1 / self.sigma) * (arx[order[:len(self.weights)]] - mean)
            self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) +
                      self.cmu * np.dot(artmp.T, np.dot(np.diag(self.weights), artmp)))
            self.B, self.D = np.linalg.eigh(self.C)
            self.BD = self.B * np.sqrt(self.D)
        return self.best_value, self.best_solution