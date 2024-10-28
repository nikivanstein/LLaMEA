import numpy as np
import random
import operator
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        # Initialize population
        population = self.generate_initial_population(self.budget, self.dim)

        for _ in range(iterations):
            # Select parents using tournament selection
            parents = self.select_parents(population, self.budget)

            # Crossover (reproduce) offspring
            offspring = self.crossover(parents)

            # Mutate offspring
            offspring = self.mutate(offspring)

            # Evaluate fitness
            fitness = [self.func(individual) for individual in offspring]

            # Select fittest individuals
            fittest_individuals = self.select_fittest(population, fitness, self.budget)

            # Replace least fit individuals with new offspring
            population = self.replace_least_fit(population, fittest_individuals, fitness, self.budget)

            # Limit population size
            population = self.limit_population(population, self.budget)

        return population

    def generate_initial_population(self, budget, dim):
        return [[random.uniform(self.search_space[0], self.search_space[1]) for _ in range(dim)] for _ in range(budget)]

    def select_parents(self, population, budget):
        # Select parents using tournament selection
        tournament_size = 3
        tournament_results = []
        for _ in range(budget):
            parent1, parent2, parent3 = random.sample(population, tournament_size)
            tournament_results.append((parent1, parent2, parent3))
        tournament_results = [pair[0] for pair in tournament_results]  # Get the best individual in each tournament
        return tournament_results

    def crossover(self, parents):
        # Crossover (reproduce) offspring
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        # Mutate offspring
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = list(individual)
            for i in range(len(individual)):
                if random.random() < 0.01:
                    mutated_individual[i] += random.uniform(-0.01, 0.01)
            mutated_offspring.append(tuple(mutated_individual))
        return mutated_offspring

    def select_fittest(self, population, fitness, budget):
        # Select fittest individuals
        fittest_individuals = []
        for _ in range(budget):
            best_individual = max(population, key=lambda individual: fitness[individual])
            fittest_individuals.append(best_individual)
        return fittest_individuals

    def replace_least_fit(self, population, fittest_individuals, fitness, budget):
        # Replace least fit individuals with new offspring
        new_population = []
        for _ in range(budget):
            individual = random.choice(population)
            if fitness[individual] < fitness[fittest_individuals[-1]]:
                new_population.append(individual)
            else:
                new_population.append(fittest_individuals[-1])
        return new_population

    def limit_population(self, population, budget):
        # Limit population size
        return population[:budget]

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using genetic programming