import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Evaluate the function using the given budget
        evaluations = np.random.randint(0, self.budget + 1, size=self.dim)
        func_evaluations = func(*evaluations)

        # Initialize the population with random solutions
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Select the fittest solutions
        self.fitness_scores = []
        for _ in range(100):
            parents = random.sample(self.population, 50)
            offspring = self.select_parents(parents, self.dim)
            self.population = self.crossover(parents, offspring)
            self.fitness_scores.append(self.evaluate_func(func_evaluations, parents, offspring))

    def select_parents(self, parents, dim):
        # Select parents using tournament selection
        tournament_size = 5
        tournament_scores = []
        for _ in range(tournament_size):
            parent1, parent2 = random.sample(parents, 2)
            score1, score2 = self.evaluate_func(func_evaluations, parent1, parent2)
            tournament_scores.append((score1, score2))
        tournament_scores.sort(key=lambda x: x[0])
        parents = [parent for score1, score2 in tournament_scores[:tournament_size] if score1 > score2]
        return parents

    def crossover(self, parents, offspring):
        # Perform crossover to generate offspring
        offspring_size = 50
        offspring = []
        for _ in range(offspring_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent1[crossover_point:], parent2[:crossover_point]))
            offspring.append(child1)
            offspring.append(child2)
        return offspring

    def evaluate_func(self, func_evaluations, parents, offspring):
        # Evaluate the function for each pair of parents and offspring
        return np.mean(func_evaluations[parents] + func_evaluations[offspring])

# Description: Evolutionary Algorithm with Adaptive Search
# Code: 