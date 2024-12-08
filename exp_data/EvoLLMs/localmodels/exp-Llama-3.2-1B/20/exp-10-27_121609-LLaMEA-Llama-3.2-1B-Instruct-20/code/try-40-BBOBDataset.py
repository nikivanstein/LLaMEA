import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class BBOBDataset(Dataset):
    def __init__(self, data, labels, func, budget):
        self.data = data
        self.labels = labels
        self.func = func
        self.budget = budget

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        func = self.func(self.data[idx])
        fitness = self.func(self.data[idx])
        return func, fitness, self.labels[idx]

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def evaluate_fitness(individual):
            fitness = objective(individual)
            self.fitnesses[individual] = fitness
            return individual

        def mutate(individual):
            bounds = bounds(individual)
            new_individual = individual.copy()
            for i in range(self.dim):
                if random.random() < 0.2:
                    new_individual[i] += np.random.uniform(-5.0, 5.0)
            return new_individual

        def crossover(parent1, parent2):
            child = parent1.copy()
            for i in range(self.dim):
                if random.random() < 0.5:
                    child[i] = parent2[i]
            return child

        def selection(population):
            return np.random.choice(len(population), self.population_size, replace=False)

        def mutation_selection(population, mutation_rate):
            return selection(population) * mutation_rate

        def mutation_crossover(population, mutation_rate):
            return mutation_selection(population, mutation_rate) * mutation_rate

        def mutation mutate(individual):
            if random.random() < 0.2:
                bounds = bounds(individual)
                new_individual = individual.copy()
                for i in range(self.dim):
                    new_individual[i] += np.random.uniform(-5.0, 5.0)
                return new_individual
            else:
                return individual

        self.population_history.append(evaluate_fitness(self.population[0]))

        for _ in range(self.budget):
            for i in range(self.population_size):
                if random.random() < 0.5:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = crossover(parent1, parent2)
                    child = mutate(child)
                    child = mutation_crossover(child, 0.2)
                    child = mutation_mutate(child)
                    self.population[i] = child
                    self.fitnesses[i] = evaluate_fitness(child)

    def get_best_individual(self):
        return self.population[np.argmax(self.fitnesses)]

# Example usage:
func = lambda x: x**2
dataset = BBOBDataset(data=[[1, 2], [3, 4], [5, 6]], labels=[1, 2, 3], func=func, budget=100)
nneo = NNEO(budget=100, dim=2)
nneo.get_best_individual()