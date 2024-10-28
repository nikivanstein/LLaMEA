import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        population_size = 100
        for _ in range(iterations):
            if _ >= self.budget:
                break
            # Select individuals using non-uniform sampling
            selection_indices = np.random.choice(self.dim, population_size, replace=False)
            selection_indices = np.argsort(selection_indices)
            selection_indices = selection_indices[:population_size // 2]
            # Select parents using tournament selection
            parents = np.random.choice(self.dim, population_size, replace=False)
            tournament_size = 5
            for _ in range(tournament_size):
                tournament_indices = np.random.choice(self.dim, tournament_size, replace=False)
                tournament_indices = np.argsort(tournament_indices)
                tournament_indices = tournament_indices[:population_size // 2]
                tournament_parent1 = parents[selection_indices[tournament_indices]]
                tournament_parent2 = parents[selection_indices[tournament_indices]]
                tournament_parent1, tournament_parent2 = self.evaluate_fitness(tournament_parent1), self.evaluate_fitness(tournament_parent2)
                parents[selection_indices[tournament_indices]] = tournament_parent1 if tournament_parent1 < tournament_parent2 else tournament_parent2
            # Select offspring using mutation and crossover
            offspring_indices = np.random.choice(self.dim, population_size, replace=False)
            for i in range(population_size):
                if i < offspring_indices.size // 2:
                    offspring_indices[i] = random.uniform(self.search_space[0], self.search_space[1])
                else:
                    parent1 = parents[offspring_indices[i - offspring_indices.size // 2]]
                    parent2 = parents[offspring_indices[i - offspring_indices.size // 2 + offspring_indices.size // 2]]
                    mutation_rate = 0.1
                    crossover_probability = 0.5
                    if random.random() < mutation_rate:
                        offspring_indices[i] = parent1 + (parent2 - parent1) * random.uniform(-mutation_rate, mutation_rate)
                    else:
                        offspring_indices[i] = parent2
            # Evaluate fitness of offspring
            offspring = []
            for i in range(population_size):
                individual = [x + random.uniform(-0.01, 0.01) for x in parents[offspring_indices[i]]]
                individual = self.evaluate_fitness(individual)
                offspring.append(individual)
            # Select best individual
            best_individual = offspring[np.argmax(offspring)]
            best_individual, best_value = self.evaluate_fitness(best_individual)
            # Update population
            population = parents + offspring
            population = population[:population_size // 2]
            # Select new parents
            selection_indices = np.random.choice(self.dim, population_size, replace=False)
            selection_indices = np.argsort(selection_indices)
            selection_indices = selection_indices[:population_size // 2]
            parents = np.random.choice(self.dim, population_size, replace=False)
            tournament_size = 5
            for _ in range(tournament_size):
                tournament_indices = np.random.choice(self.dim, tournament_size, replace=False)
                tournament_indices = np.argsort(tournament_indices)
                tournament_indices = tournament_indices[:population_size // 2]
                tournament_parent1 = parents[selection_indices[tournament_indices]]
                tournament_parent2 = parents[selection_indices[tournament_indices]]
                tournament_parent1, tournament_parent2 = self.evaluate_fitness(tournament_parent1), self.evaluate_fitness(tournament_parent2)
                parents[selection_indices[tournament_indices]] = tournament_parent1 if tournament_parent1 < tournament_parent2 else tournament_parent2
            # Update best individual
            best_individual = parents[np.argmax(offspring)]
            best_value = self.evaluate_fitness(best_individual)
        return best_individual, best_value

    def evaluate_fitness(self, individual):
        return self.func(individual)