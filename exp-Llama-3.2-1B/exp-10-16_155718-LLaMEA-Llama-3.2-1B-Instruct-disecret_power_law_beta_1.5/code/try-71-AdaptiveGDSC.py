# Description: Adaptive Black Box Optimization using Genetic Algorithm with Evolutionary Search Space Refinement
# Code:
import random
import numpy as np

class AdaptiveGDSC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None
        self.population = []

    def __call__(self, func):
        # Evaluate the function 1 time
        self.func_values[func.__name__] = func()
        
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

        # Refine the search space
        if self.search_space[0] < -0.1:
            self.search_space = (-0.1, -5.0)
        elif self.search_space[1] > 5.0:
            self.search_space = (5.0, 5.0)

        # Add the new individual to the population
        self.population.append((func.__name__, func.__code__, func.__name__, self.func_values[func.__name__]))

    def select(self, population, budget):
        # Select the best individuals
        selected = []
        for _ in range(min(budget, len(population))):
            selected.append((random.choice([i for i in population if i[1].budget > 0]), i[1].budget, i[1].func_values[i[0]]))
        return selected

    def mutate(self, selected, budget):
        # Mutate the selected individuals
        mutated = []
        for i in range(len(selected)):
            new_individual = selected[i][0]
            for j in range(len(selected[i])):
                new_individual[j] = random.uniform(self.search_space[j])
            mutated.append((new_individual, selected[i][1], selected[i][2]))
        return mutated

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func = self.func_values[individual[2]]
        return func(individual[0])

# Description: Adaptive Black Box Optimization using Genetic Algorithm with Evolutionary Search Space Refinement
# Code:
adaptive_gdsc = AdaptiveGDSC(100, 5)
adaptive_gdsc.select(adaptive_gdsc.population, 10)
adaptive_gdsc.mutate(adaptive_gdsc.population, 10)
adaptive_gdsc.evaluate_fitness(adaptive_gdsc.population[0][0])