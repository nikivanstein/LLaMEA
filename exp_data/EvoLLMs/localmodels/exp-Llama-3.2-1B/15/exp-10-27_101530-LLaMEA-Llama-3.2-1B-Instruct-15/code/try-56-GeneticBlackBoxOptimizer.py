import random
import numpy as np

class GeneticBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.elite_size = 10

    def __call__(self, func):
        # Initialize population with random points in the search space
        population = self.generate_population(self.population_size, self.search_space)

        while self.func_evaluations < self.budget:
            # Select elite individuals
            elite = self.select_elite(population, self.elite_size)

            # Evaluate fitness of elite individuals
            fitness = self.evaluate_fitness(elite, func)

            # Select parents using tournament selection
            parents = self.select_parents(population, fitness, elite, self.mutation_rate)

            # Crossover (recombination) to create offspring
            offspring = self.crossover(parents)

            # Mutate offspring to introduce diversity
            mutated_offspring = self.mutate(offspring)

            # Replace least fit individuals with offspring
            population = self.replace_least_fit(population, mutated_offspring, fitness)

            # Update best individual
            best_individual = self.select_best(population, fitness)

            # Replace least fit individuals with best individual
            population = self.replace_least_fit(population, [best_individual], fitness)

            # Increment function evaluations
            self.func_evaluations += 1

            # Check if the budget is reached
            if self.func_evaluations < self.budget:
                # If not, return the best individual found so far
                return best_individual
        # If the budget is reached, return the best individual found so far
        return best_individual

    def generate_population(self, size, space):
        return [random.uniform(space[0], space[1]) for _ in range(size)]

    def select_elite(self, population, size):
        return sorted(population, key=lambda x: x[1], reverse=True)[:size]

    def evaluate_fitness(self, individuals, func):
        return [func(individual) for individual in individuals]

    def select_parents(self, population, fitness, elite, mutation_rate):
        parents = []
        for _ in range(len(elite)):
            tournament = random.sample(population, 3)
            winner = max(tournament, key=lambda x: x[1])
            if random.random() < mutation_rate:
                winner = random.choice([winner, random.choice(population)])
            parents.append(winner)
        return parents

    def crossover(self, parents):
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def replace_least_fit(self, population, offspring, fitness):
        least_fit_individual = min(population, key=lambda x: x[1])
        return [individual for individual in offspring if individual!= least_fit_individual] + [least_fit_individual]

    def select_best(self, population, fitness):
        return max(population, key=lambda x: x[1])

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm
# Code: 