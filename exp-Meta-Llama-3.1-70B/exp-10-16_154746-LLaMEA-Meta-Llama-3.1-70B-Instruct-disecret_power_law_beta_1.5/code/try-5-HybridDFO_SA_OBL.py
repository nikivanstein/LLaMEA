import numpy as np

class HybridDFO_SA_OBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.T = 1000
        self.alpha = 0.99
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.best_fitness = np.inf
        self.worst_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.worst_fitness = -np.inf
        self.food_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.food_fitness = np.inf
        self.enemy_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.enemy_fitness = -np.inf

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
                    self.worst_fitness = self.best_fitness
                    self.worst_position = self.best_position
                    self.best_fitness = fitness
                    self.best_position = self.population[i]
                elif fitness > self.worst_fitness:
                    self.worst_fitness = fitness
                    self.worst_position = self.population[i]
                s = np.random.uniform(0, 1, self.dim)
                a = 0.1 * s
                c = 0.1 * s
                f = np.random.uniform(-1, 1, self.dim)
                x = self.population[i]
                step = (self.best_position - x) * a + (self.food_position - x) * c + f * np.abs(self.enemy_position - x)
                self.population[i] = x + step
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
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
        return self.best_position