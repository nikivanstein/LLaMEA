import numpy as np
import random

class MultiSwarmEvolution:
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
        self.swarms = [self.initialize_swarm() for _ in range(5)]

    def initialize_swarm(self):
        swarm = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            swarm.append(individual)
        return swarm

    def evaluate(self, func):
        for swarm in self.swarms:
            for individual in swarm:
                func(individual)

    def update(self):
        for swarm in self.swarms:
            for _ in range(self.swarm_size):
                for individual in swarm:
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    v1 = r1 * self.c1 * (individual - self.swarm[random.randint(0, len(self.swarms) - 1)][random.randint(0, len(self.swarm[random.randint(0, len(self.swarms) - 1)]) - 1)])
                    v2 = r2 * self.c2 * (self.swarm[random.randint(0, len(self.swarms) - 1)][random.randint(0, len(self.swarm[random.randint(0, len(self.swarms) - 1)]) - 1)] - individual)
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
        diversity = np.zeros((len(self.swarms), self.population_size, self.dim))
        for i in range(len(self.swarms)):
            for j in range(self.population_size):
                for k in range(self.population_size):
                    distance = np.linalg.norm(self.swarms[i][j] - self.swarms[i][k])
                    diversity[i, j, :] = np.append(distance, 1 - distance)
        return diversity

    def adaptive(self):
        diversity = self.diversity()
        for i in range(len(self.swarms)):
            for j in range(self.population_size):
                min_diversity = np.inf
                min_index = -1
                for k in range(self.population_size):
                    if diversity[i, j, 0] < min_diversity:
                        min_diversity = diversity[i, j, 0]
                        min_index = k
                if diversity[i, j, 0] < 0.5:
                    self.swarms[i][j] = self.swarms[i][min_index]

    def run(self, func):
        for _ in range(self.warmup):
            self.evaluate(func)
            for swarm in self.swarms:
                self.update()
                self.adaptive()
        for _ in range(self.budget - self.warmup):
            self.evaluate(func)
            for swarm in self.swarms:
                self.update()
                self.adaptive()
        return self.swarms[np.argmin([func(individual) for individual in [individual for swarm in self.swarms for individual in swarm]])]

# Example usage:
def func(x):
    return np.sum(x**2)

ms = MultiSwarmEvolution(100, 10)
result = ms(func)
print(result)