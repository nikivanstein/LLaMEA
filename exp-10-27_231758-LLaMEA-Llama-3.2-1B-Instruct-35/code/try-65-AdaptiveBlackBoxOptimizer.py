import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def genetic_algorithm(self, func, mutation_rate, crossover_rate, population_size):
        # Initialize population with random solutions
        population = self.initialize_population(func, population_size)

        while len(population) > 0:
            # Select parents using tournament selection
            parents = self.select_parents(population, 10)

            # Perform crossover
            offspring = self.crossover(parents)

            # Perform mutation
            offspring = self.mutate(offspring, mutation_rate)

            # Replace worst individuals with new offspring
            population = self.replace_worst(population, offspring)

        # Evaluate fitness of each individual
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

        # Select best individual
        best_individual = self.select_best(population, fitnesses)

        # Update parameters
        self.func_values = best_individual
        self.func_evals = len(population)
        self.func_values = np.zeros(self.dim)

        return best_individual

    def initialize_population(self, func, population_size):
        # Initialize population with random solutions
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)]

    def select_parents(self, population, num_parents):
        # Select parents using tournament selection
        winners = []
        for _ in range(num_parents):
            winner_idx = random.choice([i for i in range(len(population)) if len(population[i]) > 0])
            winner = population[winner_idx]
            num_wins = 0
            for individual in population:
                if individual is not None and np.allclose(individual, winner):
                    num_wins += 1
            winners.append((winner, num_wins))
        winners = sorted(winners, key=lambda x: x[1], reverse=True)
        return [individual for winner, num_wins in winners[:num_parents] for individual in winner]

    def crossover(self, parents):
        # Perform crossover
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = np.concatenate((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))
            offspring.append(child)
        return offspring

    def mutate(self, offspring, mutation_rate):
        # Perform mutation
        mutated_offspring = []
        for individual in offspring:
            if random.random() < mutation_rate:
                idx = random.randint(0, self.dim - 1)
                individual[idx] += np.random.uniform(-1, 1)
                if individual[idx] < -5.0:
                    individual[idx] = -5.0
                elif individual[idx] > 5.0:
                    individual[idx] = 5.0
            mutated_offspring.append(individual)
        return mutated_offspring

    def replace_worst(self, population, offspring):
        # Replace worst individuals with new offspring
        worst_individual = None
        worst_fitness = float('inf')
        for individual in population:
            if individual is not None and np.allclose(individual, worst_individual):
                fitness = self.evaluate_fitness(individual, func)
                if fitness < worst_fitness:
                    worst_individual = individual
                    worst_fitness = fitness
        for individual in offspring:
            if individual is not None and np.allclose(individual, worst_individual):
                fitness = self.evaluate_fitness(individual, func)
                if fitness < worst_fitness:
                    worst_individual = individual
                    worst_fitness = fitness
        population = [individual if np.allclose(individual, worst_individual) else worst_individual for individual in population]

    def evaluate_fitness(self, individual, func):
        # Evaluate fitness of each individual
        return func(individual)

# Description: Adaptive Black Box Optimization using Genetic Algorithm with Adaptive Crossover
# Code: 