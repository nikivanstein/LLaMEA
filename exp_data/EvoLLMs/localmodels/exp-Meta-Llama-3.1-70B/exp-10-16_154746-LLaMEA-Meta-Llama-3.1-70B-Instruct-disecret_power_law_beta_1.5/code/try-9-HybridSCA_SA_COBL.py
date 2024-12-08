import numpy as np

class HybridSCA_SA_COBL:
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
        self.chaotic_map = np.random.uniform(0, 1, self.dim)

    def opposition_based_learning(self, individual):
        opposite_individual = self.lower_bound + self.upper_bound - individual
        return opposite_individual

    def chaotic_opposition_based_learning(self, individual):
        chaotic_individual = self.lower_bound + (self.upper_bound - self.lower_bound) * self.chaotic_map
        self.chaotic_map = 4 * self.chaotic_map * (1 - self.chaotic_map)
        return chaotic_individual

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
                sine = np.abs(self.best_individual - self.population[i]) * np.sin(np.abs(r1) * np.pi)
                cosine = np.abs(self.best_individual - self.population[i]) * np.cos(np.abs(r1) * np.pi)
                if np.random.uniform(0, 1) < 0.5:
                    self.population[i] = self.best_individual - (A * sine + B * cosine) * C * D
                else:
                    self.population[i] = self.best_individual + (A * sine + B * cosine) * C * D
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
                # Apply chaotic opposition-based learning with a probability of 0.2
                if np.random.uniform(0, 1) < 0.2:
                    chaotic_individual = self.chaotic_opposition_based_learning(self.population[i])
                    chaotic_fitness = func(chaotic_individual)
                    evaluations += 1
                    if chaotic_fitness < fitness:
                        self.population[i] = chaotic_individual
        return self.best_individual