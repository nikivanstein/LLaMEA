import numpy as np
import random
from scipy.stats import norm

class ADE_SACPLR_Bayes:
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
        self.bayes_params = {'F': 0.5, 'CR': 0.5,'sigma': 0.1, 'learning_rate': 0.01, 'crossover_probability': 0.5}

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
                    x_new = self.x[r1] + self.bayes_params['F'] * (self.x[r2] - self.x[r3])
                    x_new = x_new + self.bayes_params['sigma'] * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.bayes_params['CR'] = self.bayes_params['CR'] + self.bayes_params['learning_rate'] * (self.bayes_params['crossover_probability'] - self.bayes_params['CR'])
            self.bayes_params['crossover_probability'] = max(0.1, min(1.0, self.bayes_params['CR']))
            self.bayes_params['sigma'] = self.bayes_params['sigma'] + self.bayes_params['learning_rate'] * (self.bayes_params['sigma'] - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
            # Update parameters using Bayesian optimization
            for param in ['F', 'CR','sigma', 'learning_rate', 'crossover_probability']:
                mean, std = norm.fit(self.bayes_params[param])
                self.bayes_params[param] = np.random.normal(mean, std)
        return self.best_x, self.best_fitness