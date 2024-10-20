import numpy as np
import random

class NSAMOE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.hbest = np.zeros((self.population_size, self.dim))
        self.hgbest = np.zeros(self.dim)
        self.candidate = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.harmony_size = 10
        self.pso_alpha = 0.8
        self.pso_beta = 0.4
        self.de_mutation_prob = 0.2
        self.de_crossover_prob = 0.5
        self.de_mutation_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current candidate
            values = func(self.candidate)
            # Update the pbest
            for i in range(self.population_size):
                if values[i] < self.pbest[i, :]:
                    self.pbest[i, :] = self.candidate[i, :]
            # Update the gbest
            if np.min(values) < self.gbest:
                self.gbest = np.min(values)
            # Select the fittest individuals
            fitness = np.min(values, axis=1)
            indices = np.argsort(fitness)
            self.hbest[:, :] = self.candidate[indices[:self.harmony_size], :]
            # Update the hgbest
            if np.min(fitness[:self.harmony_size]) < self.hgbest:
                self.hgbest = np.min(fitness[:self.harmony_size])
            # Apply PSO
            self.update_pso()
            # Apply HS
            self.update_hs()
            # Apply DE
            self.update_de()
            # Update the candidate
            self.candidate = self.update_candidate()
        return self.gbest

    def update_pso(self):
        for i in range(self.population_size):
            r1 = random.random()
            r2 = random.random()
            self.candidate[i, :] += self.pso_alpha * (self.pbest[i, :] - self.candidate[i, :]) + self.pso_beta * (self.hbest[i, :] - self.candidate[i, :])

    def update_hs(self):
        for i in range(self.population_size):
            r1 = random.random()
            r2 = random.random()
            self.candidate[i, :] += r1 * (self.hbest[i, :] - self.candidate[i, :]) + r2 * (self.gbest - self.candidate[i, :])

    def update_de(self):
        for i in range(self.population_size):
            if random.random() < self.de_mutation_prob:
                j = random.randint(0, self.dim-1)
                self.candidate[i, j] = self.pbest[i, j] + np.random.uniform(-self.de_mutation_rate, self.de_mutation_rate)
            if random.random() < self.de_crossover_prob:
                k = random.randint(0, self.dim-1)
                self.candidate[i, k] = self.hbest[i, k] + np.random.uniform(-self.de_mutation_rate, self.de_mutation_rate)

    def update_candidate(self):
        for i in range(self.population_size):
            r1 = random.random()
            r2 = random.random()
            self.candidate[i, :] += r1 * (self.hbest[i, :] - self.candidate[i, :]) + r2 * (self.gbest - self.candidate[i, :])
        return self.candidate

# Example usage:
def func(x):
    return np.sum(x**2)

nsamoe = NSAMOE(budget=100, dim=10)
result = nsamoe(func)
print(result)