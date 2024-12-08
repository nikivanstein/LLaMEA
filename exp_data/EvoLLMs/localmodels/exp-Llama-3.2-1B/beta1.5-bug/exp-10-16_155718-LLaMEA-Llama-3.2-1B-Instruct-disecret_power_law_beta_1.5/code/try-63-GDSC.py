# Description: A novel metaheuristic algorithm that uses a combination of clustering and local search to optimize black box functions.
# Code: 
import numpy as np
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class GDSC:
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

    def local_search(self, func, initial_individual, budget):
        # Perform local search on the current population
        for _ in range(budget):
            # Evaluate the function at each individual
            func_values = {func.__name__: func(individual) for individual in self.func_values.values()}
            
            # Select the fittest individual
            fittest_individual = max(func_values, key=func_values.get)
            
            # Reassign the selected individual to the closest cluster center
            dist = cdist([fittest_individual], self.cluster_centers, metric='euclidean')
            index = np.argmin(dist)
            self.cluster_centers[index] = fittest_individual
            
            # Update the function values for the next iteration
            func_values[fittest_individual] = func(fittest_individual)
            
            # Evaluate the function 1 time
            func_values[fittest_individual] = func(fittest_individual)
        
        # Return the fittest individual
        return max(func_values, key=func_values.get)

# Description: A novel metaheuristic algorithm that uses a combination of clustering and local search to optimize black box functions.
# Code: 
if __name__ == "__main__":
    # Create an instance of the GDSC algorithm
    gdsc = GDSC(10, 5)
    
    # Define a black box function
    def func(x):
        return np.sin(x) + 0.5 * np.cos(x)
    
    # Initialize the population with random individuals
    individuals = np.random.uniform(-5.0, 5.0, (10, 5))
    
    # Perform the local search
    gdsc.local_search(func, individuals, 10)
    
    # Print the fittest individual
    print(gdsc.cluster_centers)