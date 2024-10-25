import numpy as np
import random

class IAE:
    def __init__(self, budget, dim, alpha=0.1, gamma=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.gamma = gamma
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.iterations = 0

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Initialize the grid with random values
        self.grid[:, :, 0] = np.random.rand(self.dim)
        
        # Generate new hypergrids by perturbing the current location
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x
            
            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i] = x
            self.budgets[i] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

    def iterative_adaptation(self, func):
        # Initialize the current solution and its fitness
        x = np.random.rand(self.dim)
        y = np.array([func(x)])
        
        # Initialize the best solution and its fitness
        best_x = x
        best_y = y
        
        # Initialize the number of iterations
        iterations = 0
        
        while iterations < self.budget:
            # Generate a new solution by perturbing the current solution
            x = self.grid[:, :, iterations % self.grid_size].copy()
            y = np.array([func(x)])
            
            # Update the best solution and its fitness
            best_x = x
            best_y = y
            
            # Update the current solution and its fitness
            x = np.random.rand(self.dim)
            y = np.array([func(x)])
            
            # Increment the number of iterations
            iterations += 1
            
            # If the current solution is better than the best solution, update the best solution
            if y > best_y:
                best_x = x
                best_y = y
        
        # Return the best solution
        return best_x, best_y

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an IAE instance with 10 budget evaluations and 3 dimensions
    iae = IAE(10, 3)
    
    # Optimize the function using IAE
    best_x, best_y = iae.iterative_adaptation(func)
    print("Optimal solution:", best_x)
    print("Optimal fitness:", best_y)