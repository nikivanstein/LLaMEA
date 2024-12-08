import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.nn = NeuralNetwork(self.dim)

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
            best_individual = self.population[np.argmax(fitness_scores)]
            new_individual = self.nn(individual)
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(best_individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def select_best(self, func, population, budget):
        selected_population = []
        for _ in range(budget):
            fitness_scores = [fitness(individual) for individual in population]
            best_individual = population[np.argmax(fitness_scores)]
            new_individual = self.nn(best_individual)
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(best_individual):
                selected_population.append(new_individual)
            else:
                selected_population.append(best_individual)
        return selected_population

    def mutate_population(self, population, mutation_rate):
        for i in range(len(population)):
            if random.random() < mutation_rate:
                self.population[i] = self.mutate(population[i])
        return population

    def update(self, func, population, budget):
        selected_population = self.select_best(func, population, budget)
        new_population = self.mutate_population(population, 0.1)
        return selected_population, new_population

# One-line description with the main idea
# Evolutionary Black Box Optimization using Neural Network
# 
# This algorithm optimizes a black box function using a neural network to approximate the function, 
# and then selects the best individual from the population based on the fitness scores, 
# and mutates the population to refine the strategy.

# Code