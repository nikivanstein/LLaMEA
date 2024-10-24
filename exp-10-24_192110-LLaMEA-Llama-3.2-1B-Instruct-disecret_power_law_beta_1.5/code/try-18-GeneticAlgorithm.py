import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_population()

    def generate_population(self):
        population = []
        for _ in range(100):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def __call__(self, func):
        best_individual = None
        best_fitness = -np.inf
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents(population)
            # Crossover (reproduce) offspring
            offspring = self.crossover(parents)
            # Mutate offspring
            offspring = self.mutate(offspring)
            # Evaluate fitness of offspring
            fitness = self.evaluate_fitness(offspring, func)
            # Replace worst individual with new offspring
            if fitness > best_fitness:
                best_individual = offspring
                best_fitness = fitness
        return best_individual

    def select_parents(self, population):
        # Select parents using tournament selection
        tournament_size = 3
        winners = []
        for _ in range(tournament_size):
            parent = random.choice(population)
            winners.append(parent)
            while len(winners) < tournament_size:
                winner = random.choice(population)
                if winner not in winners:
                    winners.append(winner)
        return winners

    def crossover(self, parents):
        # Crossover (reproduce) offspring
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        # Mutate offspring
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual + random.uniform(-1, 1) / 10
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def evaluate_fitness(self, offspring, func):
        # Evaluate fitness of offspring
        fitness = 0
        for individual in offspring:
            func(individual)
            fitness += 1
        return fitness

# Description: Genetic Algorithm for Black Box Optimization
# Code: 
# ```python
ga = GeneticAlgorithm(100, 10)  # 100 generations, 10 dimensions
ga.population = np.random.uniform(-5.0, 5.0, (100, 10))  # initialize population
best_individual = ga(__call__(ga.func))  # optimize the function
print("Best Individual:", best_individual)
print("Best Fitness:", ga.evaluate_fitness(best_individual, ga.func))  # print the best fitness