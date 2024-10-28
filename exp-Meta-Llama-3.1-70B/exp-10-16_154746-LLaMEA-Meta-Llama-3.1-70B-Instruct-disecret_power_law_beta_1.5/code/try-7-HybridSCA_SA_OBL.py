import numpy as np

class HybridSCA_SA_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.T = 1000
        self.alpha = 0.99
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.best_fitness = np.inf

    def opposition_based_learning(self, individual):
        opposite_individual = self.lower_bound + self.upper_bound - individual
        return opposite_individual

    def levy_flight(self, individual):
        levy_flight = np.random.normal(0, 1, self.dim) / np.random.normal(0, 1, self.dim)
        levy_flight = individual + levy_flight
        return levy_flight

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.population[i])
                evaluations += 1
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = self.population[i]
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                r3 = np.random.uniform(0, 1, self.dim)
                r4 = np.random.uniform(0, 1, self.dim)
                A = 2 * r1 - 1
                B = np.abs(r2)
                C = 2 * r3 - 1
                D = np.abs(r4)
                if r1 < 0.5:
                    X1 = self.best_individual + r1 * np.abs(C * self.best_individual - self.population[i]) * np.sin(np.pi * r2)
                else:
                    X1 = self.best_individual - r1 * np.abs(C * self.best_individual - self.population[i]) * np.sin(np.pi * r2)
                if r3 < 0.5:
                    X2 = self.best_individual + r3 * np.abs(D * self.best_individual - self.population[i]) * np.cos(np.pi * r4)
                else:
                    X2 = self.best_individual - r3 * np.abs(D * self.best_individual - self.population[i]) * np.cos(np.pi * r4)
                self.population[i] = (X1 + X2) / 2
                levy_flight = self.levy_flight(self.population[i])
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
        return self.best_individual