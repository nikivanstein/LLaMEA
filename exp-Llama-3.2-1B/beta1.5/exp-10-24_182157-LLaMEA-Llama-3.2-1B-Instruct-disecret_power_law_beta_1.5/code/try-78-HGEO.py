# Import necessary libraries
import numpy as np
import random
import math

# Define a function to calculate the probability of convergence
def prob_convergence(new_individual, old_individual, new_fitness, old_fitness):
    return (new_fitness - old_fitness) / (math.sqrt((new_fitness - old_fitness)**2 + (1 / (new_individual + old_individual))**2))

# Define a function to perform mutation on an individual
def mutate(individual, budget):
    new_individual = individual.copy()
    for _ in range(budget):
        if random.random() < 0.5:
            new_individual[random.randint(0, len(individual) - 1)] += random.uniform(-1, 1)
    return new_individual

# Define the HGEO class with the updated __init__ method
class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []

    # Define the __call__ method to optimize the function
    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x
            
            # Generate new hypergrids by perturbing the current location
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i + 1] = x
            self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i + 1])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

# Define the example function to optimize
def func(x):
    return x[0]**2 + x[1]**2

# Create an HGEO instance with 10 budget evaluations and 3 dimensions
hgeo = HGEO(10, 3)

# Optimize the function using HGEO
optimal_x = hgeo(func)
print("Optimal solution:", optimal_x)

# Define the example function to optimize
def func(x):
    return x[0]**2 + x[1]**2

# Create an HGEO instance with 10 budget evaluations and 3 dimensions
hgeo2 = HGEO(10, 3)

# Optimize the function using HGEO2
optimal_x2 = hgeo2(func)
print("Optimal solution:", optimal_x2)

# Define the HGEO class with the updated __init__ method
class HGEO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []

    # Define the __call__ method to optimize the function
    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x
            
            # Generate new hypergrids by perturbing the current location
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i + 1] = x
            self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i + 1])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

# Optimize the function using HGEO2
optimal_x2 = HGEO2(10, 3)(func)
print("Optimal solution:", optimal_x2)