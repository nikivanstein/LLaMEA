import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import minimize

class HybridADE_SACPLR_GPO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.crossover_probability = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.gp_model = None
        self.gp_noise = 0.1

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == idx or r2 == idx or r3 == idx:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1] + self.F * (self.x[r2] - self.x[r3])
                    x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
            # Gaussian Process Optimization
            if self.gp_model is None:
                self.gp_model = self.train_gp(self.x, self.fitness)
            x_gp = self.sample_gp(self.gp_model, self.lower_bound, self.upper_bound, self.dim)
            y_gp = func(x_gp)
            if y_gp < self.best_fitness:
                self.best_fitness = y_gp
                self.best_x = x_gp
        return self.best_x, self.best_fitness

    def train_gp(self, x, y):
        gp_model = {'mean': lambda x: np.zeros(self.dim), 'cov': lambda x, x2: np.eye(self.dim)}
        gp_model['mean'] = lambda x: np.zeros(self.dim)
        gp_model['cov'] = lambda x, x2: np.eye(self.dim)
        return gp_model

    def sample_gp(self, gp_model, lower_bound, upper_bound, dim):
        x_gp = np.zeros(dim)
        for i in range(dim):
            x_gp[i] = np.random.uniform(lower_bound, upper_bound)
        return x_gp

    def optimize_gp(self, gp_model, lower_bound, upper_bound, dim):
        res = minimize(lambda x: -gp_model['mean'](x), x0=np.zeros(dim), method='SLSQP', bounds=[(lower_bound, upper_bound)]*dim)
        return res.x