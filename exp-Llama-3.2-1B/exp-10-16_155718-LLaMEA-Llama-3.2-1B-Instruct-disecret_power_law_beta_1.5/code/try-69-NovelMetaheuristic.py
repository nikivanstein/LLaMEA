import numpy as np
import random

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None
        self.population = []

    def __call__(self, func, population, logger):
        # Evaluate the function for each individual in the population
        for individual in population:
            # Evaluate the function 1 time
            self.func_values[func.__name__] = func(individual)
        
        # Initialize the cluster centers randomly
        if self.cluster_centers is None:
            self.cluster_centers = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
        
        # Assign each sample to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)
        
        # Update the function values for the next iteration
        for i in range(self.budget):
            new_centers = self.cluster_centers.copy()
            for j in range(self.dim):
                for k in range(self.dim):
                    new_centers[j, k] += 0.1 * (self.cluster_centers[j, k] - self.cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (self.cluster_centers[j, k] - self.cluster_centers[j, k] ** 2))
            self.cluster_centers = new_centers
        
        # Reassign each sample to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)
        
        # Evaluate the function 1 time
        self.func_values[func.__name__] = func()

    def select_individual(self, problem):
        # Select an individual from the population
        # Use a probability of selection based on the fitness
        fitness = self.func_values[problem.func.__name__]
        selection_prob = 1 / fitness
        selected_individual = random.choices(list(self.population), weights=selection_prob)[0]
        
        # Refine the strategy by changing the individual's line
        if self.cluster_centers is None:
            self.cluster_centers = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
        
        # Reassign the selected individual to the closest cluster center
        selected_individual = np.array([self.cluster_centers])
        for sample in problem.func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(problem.func(self.func_values[sample]) - selected_individual, axis=1)
            selected_individual = np.argmin(dist, axis=0)
        
        # Update the selected individual's fitness
        fitness = problem.func(self.func_values[selected_individual])
        self.func_values[selected_individual] = fitness
        
        return selected_individual

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        # Use a probability of mutation based on the fitness
        fitness = self.func_values[individual]
        mutation_prob = 1 / fitness
        if random.random() < mutation_prob:
            # Swap the two elements
            individual = np.array([self.cluster_centers])
            for i in range(self.dim):
                for j in range(self.dim):
                    if random.random() < mutation_prob:
                        individual[i, j] = self.cluster_centers[i, j]
            # Update the individual's fitness
            fitness = self.func_values[individual]
            self.func_values[individual] = fitness
        
        return individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 