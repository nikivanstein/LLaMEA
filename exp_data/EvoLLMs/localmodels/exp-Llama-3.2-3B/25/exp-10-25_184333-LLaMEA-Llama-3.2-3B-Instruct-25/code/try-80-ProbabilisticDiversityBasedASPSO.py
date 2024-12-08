import numpy as np
import random

class ProbabilisticDiversityBasedASPSO:
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
        self.refine_probability = 0.25

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
                if random.random() < self.refine_probability:
                    individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def diversity(self):
        diversity = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.population_size):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                diversity[i, :] = np.append(distance, 1 - distance)
        return diversity

    def adaptive(self):
        diversity = self.diversity()
        for i in range(self.population_size):
            min_diversity = np.inf
            min_index = -1
            for j in range(self.population_size):
                if diversity[j, 0] < min_diversity:
                    min_diversity = diversity[j, 0]
                    min_index = j
            if diversity[min_index, 0] < 0.5:
                self.population[i] = self.population[min_index]

    def refine(self):
        new_population = []
        for individual in self.population:
            r1 = np.random.uniform(0, 1)
            if r1 < self.refine_probability:
                new_individual = individual + np.random.uniform(-0.1, 0.1, self.dim)
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                new_population.append(new_individual)
            else:
                new_population.append(individual)
        self.population = new_population

    def run(self, func):
        for _ in range(self.warmup):
            self.evaluate(func)
            self.update()
            self.adaptive()
            self.refine()
        for _ in range(self.budget - self.warmup):
            self.evaluate(func)
            self.update()
            self.adaptive()
            self.refine()
        return self.population[np.argmin([func(individual) for individual in self.population])]

# Example usage:
def func(x):
    return np.sum(x**2)

aspo = ProbabilisticDiversityBasedASPSO(100, 10)
result = aspo(func)
print(result)