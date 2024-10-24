import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.search_space = [-5.0, 5.0]  # Define the search space

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            # Generate new hypergrids by perturbing the current location
            x = np.random.rand(self.dim)
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i] = x + np.random.uniform(-self.search_space[1], self.search_space[1], self.dim)
            
            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i] = x
            self.budgets[i] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        for j in range(self.grid_size):
            for k in range(self.grid_size):
                self.grid[:, :, -1] = x + np.random.uniform(-self.search_space[1], self.search_space[1], self.dim)
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

# One-line description with main idea
# HGEO: Hypergrid Genetic Optimization Algorithm
# Code: 