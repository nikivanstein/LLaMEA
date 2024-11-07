import numpy as np
import random

class MDAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_score = -np.inf

    def initialize_population(self):
        population = []
        for _ in range(self.budget):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        scores = []
        for individual in self.population:
            score = func(individual)
            scores.append(score)
            if score > self.best_score:
                self.best_score = score
                self.best_solution = individual
        return scores

    def mutate(self, individual):
        mutation_rate = 0.1
        for i in range(self.dim):
            if random.random() < mutation_rate:
                individual[i] += np.random.uniform(-1.0, 1.0)
                individual[i] = max(-5.0, min(5.0, individual[i]))
        return individual

    def adapt(self, scores):
        population = self.population
        for i in range(self.budget):
            if scores[i] < self.best_score:
                population[i] = self.mutate(population[i])
        return population

    def __call__(self, func):
        scores = self.evaluate(func)
        self.population = self.adapt(scores)
        return self.best_solution, self.best_score

# Example usage:
budget = 100
dim = 10
mdaea = MDAEA(budget, dim)
best_solution, best_score = mdaeas(func)
print(f"Best solution: {best_solution}")
print(f"Best score: {best_score}")