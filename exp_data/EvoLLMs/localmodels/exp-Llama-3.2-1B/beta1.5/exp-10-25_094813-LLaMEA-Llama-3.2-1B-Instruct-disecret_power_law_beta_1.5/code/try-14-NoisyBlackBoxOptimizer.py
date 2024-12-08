import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Gradient descent with hierarchical clustering for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
                    if self.current_dim == 0:
                        # Gradient descent without hierarchical clustering
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical clustering to select the best function to optimize
                        cluster_labels = np.argpartition(func, self.current_dim)[-1]
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

def evaluate_fitness(individual, func, budget):
    # Evaluate the fitness of the individual using the given function
    fitness = np.array([func(individual[i]) for i in range(len(individual))])
    return fitness

def mutate(individual, func, budget):
    # Mutate the individual by randomly changing one element
    mutated_individual = individual.copy()
    mutated_individual[ np.random.randint(len(individual))] = np.random.uniform(-5.0, 5.0)
    return mutated_individual

def selection(individuals, func, budget):
    # Select the fittest individuals using the given function
    fitnesses = evaluate_fitness(individuals, func, budget)
    selected_individuals = individuals[np.argsort(fitnesses)]
    return selected_individuals

def genetic_algorithm(func, budget, dim, max_iter=1000):
    # Initialize the population using the given function
    population = [func(np.random.uniform(-5.0, 5.0, dim)) for _ in range(100)]

    for _ in range(max_iter):
        # Evaluate the fitness of each individual in the population
        fitnesses = evaluate_fitness(population, func, budget)

        # Select the fittest individuals
        selected_individuals = selection(population, func, budget)

        # Mutate the selected individuals
        mutated_individuals = [mutate(individual, func, budget) for individual in selected_individuals]

        # Replace the least fit individuals with the mutated ones
        population = [individual for individual in population if individual in mutated_individuals]

        # Update the current dimension
        self.current_dim += 1

    return population

# Run the genetic algorithm
population = genetic_algorithm(func, budget, dim)

# Print the results
print("NoisyBlackBoxOptimizer:", population)