import numpy as np
import random

class AdaptiveDynamicEVA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.swarm_size = 10
        self.warmup = 10
        self.c1 = 1.5
        self.c2 = 1.5
        self.rho = 0.99
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.initialize_population()
        self.diversity = np.zeros((self.population_size, self.dim))
        self.adaptation_rate = 0.25

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        for individual in self.population:
            func(individual)

    def update(self):
        for _ in range(self.swarm_size):
            for individual in self.population:
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                v1 = r1 * self.c1 * (individual - self.population[random.randint(0, self.population_size - 1)])
                v2 = r2 * self.c2 * (self.population[random.randint(0, self.population_size - 1)] - individual)
                individual += v1 + v2
                if individual[0] < self.lower_bound:
                    individual[0] = self.lower_bound
                if individual[0] > self.upper_bound:
                    individual[0] = self.upper_bound
                if individual[1] < self.lower_bound:
                    individual[1] = self.lower_bound
                if individual[1] > self.upper_bound:
                    individual[1] = self.upper_bound
                if random.random() < 0.2:
                    individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def diversity(self):
        for i in range(self.population_size):
            for j in range(self.population_size):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                self.diversity[i, :] = np.append(distance, 1 - distance)

    def adaptation(self):
        for i in range(self.population_size):
            if random.random() < self.adaptation_rate:
                min_diversity = np.inf
                min_index = -1
                for j in range(self.population_size):
                    if self.diversity[j, 0] < min_diversity:
                        min_diversity = self.diversity[j, 0]
                        min_index = j
                if self.diversity[min_index, 0] < 0.5:
                    self.population[i] = self.population[min_index]

    def run(self, func):
        for _ in range(self.warmup):
            self.evaluate(func)
            self.update()
            self.adaptation()
        for _ in range(self.budget - self.warmup):
            self.evaluate(func)
            self.update()
            self.adaptation()
        return self.population[np.argmin([func(individual) for individual in self.population])]

# Example usage:
def func(x):
    return np.sum(x**2)

eva = AdaptiveDynamicEVA(100, 10)
result = eva(func)
print(result)