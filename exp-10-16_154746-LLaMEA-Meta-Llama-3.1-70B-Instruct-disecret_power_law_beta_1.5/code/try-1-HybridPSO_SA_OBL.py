import numpy as np

class HybridPSO_SA_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.4
        self.c2 = 1.4
        self.w = 0.9
        self.T = 1000
        self.alpha = 0.99
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.pbest = self.population.copy()
        self.pbest_fitness = np.inf * np.ones(self.population_size)
        self.gbest = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.gbest_fitness = np.inf

    def opposition_based_learning(self, individual):
        opposite_individual = self.lower_bound + self.upper_bound - individual
        return opposite_individual

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.population[i])
                evaluations += 1
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.population[i]
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = self.population[i]
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * np.random.uniform(0, 1, self.dim) * (self.pbest[i] - self.population[i]) + self.c2 * np.random.uniform(0, 1, self.dim) * (self.gbest - self.population[i])
                levy_flight = np.random.normal(0, 1, self.dim) / np.random.normal(0, 1, self.dim)
                levy_flight = self.population[i] + levy_flight
                levy_fitness = func(levy_flight)
                evaluations += 1
                if levy_fitness < fitness:
                    self.population[i] = levy_flight
                else:
                    delta = levy_fitness - fitness
                    prob = np.exp(-delta / self.T)
                    self.T *= self.alpha
                    if np.random.uniform(0, 1) < prob:
                        self.population[i] = levy_flight
                # Apply opposition-based learning with a probability of 0.2
                if np.random.uniform(0, 1) < 0.2:
                    opposite_individual = self.opposition_based_learning(self.population[i])
                    opposite_fitness = func(opposite_individual)
                    evaluations += 1
                    if opposite_fitness < fitness:
                        self.population[i] = opposite_individual
        return self.gbest