import random
import numpy as np
from scipy.optimize import minimize

class EMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None

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

    def update(self, func):
        # Update the search space and cluster centers using stochastic gradient clustering
        new_search_space = (-5.0, 5.0)
        new_cluster_centers = np.random.uniform(new_search_space[0], new_search_space[1], (self.dim, self.dim))
        
        # Assign each sample to the closest cluster center
        new_cluster_centers = np.array([new_cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - new_cluster_centers, axis=1)
            new_cluster_centers = np.argmin(dist, axis=0)
        
        # Update the search space and cluster centers using gradient descent
        new_search_space = (-5.0, 5.0)
        new_cluster_centers = np.array([new_cluster_centers])
        for j in range(self.dim):
            for k in range(self.dim):
                new_cluster_centers[j, k] += 0.1 * (new_cluster_centers[j, k] - new_cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (new_cluster_centers[j, k] - new_cluster_centers[j, k] ** 2))
        
        # Reassign each sample to the closest cluster center
        new_cluster_centers = np.array([new_cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - new_cluster_centers, axis=1)
            new_cluster_centers = np.argmin(dist, axis=0)
        
        # Update the search space and cluster centers using gradient descent
        new_search_space = (-5.0, 5.0)
        new_cluster_centers = np.array([new_cluster_centers])
        for j in range(self.dim):
            for k in range(self.dim):
                new_cluster_centers[j, k] += 0.1 * (new_cluster_centers[j, k] - new_cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (new_cluster_centers[j, k] - new_cluster_centers[j, k] ** 2))
        
        # Update the search space and cluster centers using stochastic gradient clustering
        new_cluster_centers = np.array([new_cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - new_cluster_centers, axis=1)
            new_cluster_centers = np.argmin(dist, axis=0)
        
        # Reassign each sample to the closest cluster center
        new_cluster_centers = np.array([new_cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - new_cluster_centers, axis=1)
            new_cluster_centers = np.argmin(dist, axis=0)
        
        # Evaluate the function 1 time
        self.func_values[func.__name__] = func()

# Example usage
ems = EMS(1000, 10)
func = lambda x: np.sin(x)
ems.update(func)
print(ems.func_values[func.__name__])