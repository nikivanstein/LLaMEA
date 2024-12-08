import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def __call__(self, func):
        def evaluate(individual):
            func(individual)
            return individual

        def fitness(individual):
            return np.linalg.norm(evaluate(individual) - 0)

        while len(self.population) < self.budget:
            for individual in self.population:
                if len(self.population) == self.budget:
                    break
                fitness_value = fitness(individual)
                if fitness_value > self.budget:
                    break
            else:
                self.population.append(random.uniform(-5.0, 5.0))
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            while len(self.population) >= self.budget:
                individual = self.population[random.randint(0, self.population_size - 1)]
            self.population.append(individual)

        best_individual = min(self.population, key=fitness)
        return best_individual

    def select(self, individual, probabilities):
        return [individual * p for individual, p in zip(self.population, probabilities)]

    def breed(self, parents):
        children = []
        while len(children) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = self.breed_single(parent1, parent2)
            if len(self.population) >= self.budget:
                child = self.select(child, probabilities)
            children.append(child)
        return children

    def breed_single(self, parent1, parent2):
        crossover_point = random.randint(1, self.dim)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return np.concatenate((child1, child2))

    def mutate(self, individual):
        if random.random() < 0.01:
            crossover_point = random.randint(1, self.dim)
            child = individual[:crossover_point] + individual[crossover_point + 1:]
            return np.concatenate((child, individual))
        return individual

    def __str__(self):
        return "Genetic Algorithm with Adaptation"

# Description: Evolutionary Optimization using Genetic Algorithm with Adaptation
# Code: 