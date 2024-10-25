import numpy as np

class HybridGWO_SA_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.T = 1000
        self.alpha = 0.99
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.alpha_wolf = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.beta_wolf = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.delta_wolf = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.alpha_fitness = np.inf
        self.beta_fitness = np.inf
        self.delta_fitness = np.inf

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
                if fitness < self.alpha_fitness:
                    self.delta_fitness = self.beta_fitness
                    self.delta_wolf = self.beta_wolf
                    self.beta_fitness = self.alpha_fitness
                    self.beta_wolf = self.alpha_wolf
                    self.alpha_fitness = fitness
                    self.alpha_wolf = self.population[i]
                elif fitness < self.beta_fitness:
                    self.delta_fitness = self.beta_fitness
                    self.delta_wolf = self.beta_wolf
                    self.beta_fitness = fitness
                    self.beta_wolf = self.population[i]
                elif fitness < self.delta_fitness:
                    self.delta_fitness = fitness
                    self.delta_wolf = self.population[i]
                a = 2 - evaluations / self.budget
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.alpha_wolf - self.population[i])
                X1 = self.alpha_wolf - A * D
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.beta_wolf - self.population[i])
                X2 = self.beta_wolf - A * D
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.delta_wolf - self.population[i])
                X3 = self.delta_wolf - A * D
                self.population[i] = (X1 + X2 + X3) / 3
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
        return self.alpha_wolf