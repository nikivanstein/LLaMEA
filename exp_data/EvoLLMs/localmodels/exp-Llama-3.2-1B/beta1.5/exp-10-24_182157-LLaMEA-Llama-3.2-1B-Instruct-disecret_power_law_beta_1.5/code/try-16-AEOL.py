import numpy as np
import random

class AEOL:
    def __init__(self, budget, dim, grid_size=10):
        self.budget = budget
        self.dim = dim
        self.grid_size = grid_size
        self.grid = np.zeros((dim, grid_size, grid_size))
        self.hypergrids = []
        self.grid_size_step = 1
        self.grid_size_step_factor = 0.1

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x
            
            # Generate new hypergrids by perturbing the current location
            new_grid = self.grid.copy()
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    new_grid[:, :, i + 1] = new_grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at each new location
            new_y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i + 1] = x
            self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(new_grid)
        
        # Evaluate the function at the last hypergrid location
        new_x = np.random.rand(self.dim)
        new_y = np.array([f(new_x) for new_x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(new_y)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an AEOL instance with 10 budget evaluations and 3 dimensions
    aeol = AEOL(10, 3)
    
    # Optimize the function using AEOL
    optimal_x = aeol(func)
    print("Optimal solution:", optimal_x)