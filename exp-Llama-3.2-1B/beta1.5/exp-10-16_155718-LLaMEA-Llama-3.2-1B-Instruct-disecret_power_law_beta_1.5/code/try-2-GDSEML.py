import random
import numpy as np

class GDSEML:
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

    def mutate(self, individual):
        # Randomly swap two genes
        j, k = random.sample(range(self.dim), 2)
        individual[j], individual[k] = individual[k], individual[j]
        
        # Reassign the new individual to the closest cluster center
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(individual[sample]) - self.cluster_centers, axis=1)
            new_center = np.argmin(dist, axis=0)
            individual[sample] = func(individual[sample])
            individual[new_center] = func(individual[new_center])
        
        # Reassign each sample to the closest cluster center
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(individual[sample]) - self.cluster_centers, axis=1)
            new_center = np.argmin(dist, axis=0)
            individual[sample] = func(individual[sample])
            individual[new_center] = func(individual[new_center])
        
        return individual

    def initialize_single(self, func):
        # Initialize the cluster centers randomly
        if self.cluster_centers is None:
            self.cluster_centers = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
        
        return self.cluster_centers

# Example usage
problem = RealSingleObjectiveProblem(5, "Sphere")
gdseml = GDSEML(10, 5)

# Initialize the problem
problem.initialize_single(gdseml.initialize_single)

# Run the algorithm
for _ in range(100):
    gdseml(gdseml, problem.func)