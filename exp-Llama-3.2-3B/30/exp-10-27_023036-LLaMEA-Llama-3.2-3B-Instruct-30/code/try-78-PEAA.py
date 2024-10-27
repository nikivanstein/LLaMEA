import numpy as np
import random

class PEAA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.score = self.evaluate_population()

    def initialize_population(self):
        population = []
        for _ in range(self.budget):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate_population(self):
        scores = []
        for individual in self.population:
            score = self.func(individual)
            scores.append(score)
        return scores

    def func(self, x):
        # Example black box function
        return np.sum(x**2)

    def __call__(self, func):
        for _ in range(self.budget):
            if self.budget > 0:
                # Select parents using tournament selection
                parents = self.select_parents()
                # Crossover and mutation
                offspring = self.crossover_and_mutate(parents)
                # Replace the worst individual
                self.replace_worst(offspring)
                # Update the score
                self.score = self.evaluate_population()
            else:
                break

    def select_parents(self):
        tournament_size = 5
        parents = []
        for _ in range(tournament_size):
            individual = random.choice(self.population)
            parents.append(individual)
        return parents

    def crossover_and_mutate(self, parents):
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = self.mutate(self.crossover(parent1, parent2))
            offspring.append(child)
        return offspring

    def crossover(self, parent1, parent2):
        # Simple crossover
        child = (parent1 + parent2) / 2
        return child

    def mutate(self, individual):
        # Simple mutation
        mutation_rate = 0.1
        if np.random.rand() < mutation_rate:
            individual[random.randint(0, self.dim-1)] += np.random.uniform(-1.0, 1.0)
        return individual

    def replace_worst(self, offspring):
        worst_index = np.argmin(self.score)
        self.population[worst_index] = offspring[worst_index]

# Example usage:
peaa = PEAA(100, 5)
peaa('func')