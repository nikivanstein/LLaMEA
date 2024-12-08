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
        self.mutation_rate = 0.2
        self.mutation_probability = 0.2
        self.crossover_probability = 0.5
        self.harmony_crossover_probability = 0.7

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
            # Apply NS
            self.update_ns()
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

    def update_ns(self):
        for i in range(self.population_size):
            r1 = random.random()
            r2 = random.random()
            self.candidate[i, :] += r1 * (self.pbest[i, :] - self.candidate[i, :]) + r2 * (self.hgbest - self.candidate[i, :])

    def update_candidate(self):
        new_candidate = np.zeros(self.dim)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                new_candidate += self.mutation(new_candidate)
            if random.random() < self.crossover_probability:
                new_candidate = self.crossover(new_candidate, self.candidate[i, :])
            if random.random() < self.harmony_crossover_probability:
                new_candidate = self.harmony_crossover(new_candidate, self.candidate[i, :])
        return new_candidate

    def mutation(self, individual):
        mutated_individual = individual
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                mutated_individual[i] += np.random.uniform(-1, 1)
        return mutated_individual

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.dim-1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(crossover_point, self.dim):
            child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def harmony_crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.dim-1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(crossover_point, self.dim):
            child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

# Example usage:
def func(x):
    return np.sum(x**2)

nsamoe = NSAMOE(budget=100, dim=10)
result = nsamoe(func)
print(result)