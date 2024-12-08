import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

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
            # Genetic Algorithm with Hierarchical Clustering for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform genetic algorithm with hierarchical clustering for efficient exploration-ejection
                    # Initialize population with random individuals
                    population = np.array([func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)])
                    # Evolve population using genetic algorithm
                    while self.budget > 0 and self.current_dim < self.dim:
                        # Select parents using hierarchical clustering
                        parents = np.array([population[np.argpartition(func, self.current_dim) == cluster_labels] for cluster_labels in range(self.dim)])
                        # Crossover (reproduce) parents to create offspring
                        offspring = np.array([np.concatenate((parent1, parent2)) for parent1, parent2 in zip(parents, parents[1:])])
                        # Mutate offspring to introduce genetic variation
                        offspring = np.array([np.random.uniform(-5.0, 5.0, self.dim) for individual in offspring])
                        # Replace worst individual in population with worst individual in offspring
                        worst_index = np.argmin(population)
                        population[worst_index] = offspring[worst_index]
                        # Update current dimension
                        self.current_dim += 1
                        # Reduce budget
                        self.budget -= 1
                    # Select best individual from population
                    best_individual = np.array([func(individual) for individual in population])
                    # Evaluate fitness of best individual
                    best_fitness = np.array([func(individual) for individual in best_individual])
                    # Update current individual
                    self.func = best_individual
                    return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

# One-line description with main idea
# Hierarchical Black Box Optimization using Genetic Algorithm with Hierarchical Clustering
# to solve black box optimization problems with a wide range of tasks, evaluated on the BBOB test suite of 24 noiseless functions