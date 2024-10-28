import numpy as np
import random

class EvolutionaryDESwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.swarm_size = 5
        self.swarm = self.initialize_swarm()

    def initialize_swarm(self):
        return np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.swarm)
        self.swarm = self.select_swarm(fitnesses)
        self.swarm = self.crossover(self.swarm)
        self.swarm = self.mutate(self.swarm)

    def select_swarm(self, fitnesses):
        fitnesses = np.array(fitnesses)
        swarm = np.array([self.swarm[np.argsort(fitnesses)[:int(self.swarm_size/2)]]])
        return swarm

    def crossover(self, swarm):
        offspring = np.zeros((self.swarm_size, self.dim))
        for i in range(self.swarm_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(swarm)
                parent2 = random.choice(swarm)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, swarm):
        mutated_swarm = np.copy(swarm)
        for i in range(self.swarm_size):
            if random.random() < self.mutation_probability:
                mutated_swarm[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_swarm

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.swarm, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

evds = EvolutionaryDESwarm(budget=100, dim=10)
optimal_solution = evds(func)
print(optimal_solution)