import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.current_budget = 0
        self.current_hypergrid = None

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            # Generate new hypergrids by perturbing the current location
            if self.current_hypergrid is not None:
                new_x = self.current_hypergrid + np.random.uniform(-1, 1, self.dim)
                new_grid = self.current_hypergrid
                for j in range(self.grid_size):
                    for k in range(self.grid_size):
                        new_grid[:, :, j] = new_grid[:, :, k] + np.random.uniform(-1, 1, self.dim)
                
                # Evaluate the function at the new location
                new_y = np.array([f(x) for x in new_x])
                
                # Update the hypergrid and its budget
                self.current_hypergrid = new_grid
                self.current_budget = min(self.current_budget + 1, self.budget)
                
                # Add the new hypergrid to the list of hypergrids
                self.hypergrids.append(self.current_hypergrid)
            else:
                # Evaluate the function at the current location
                new_x = np.random.rand(self.dim)
                new_grid = self.current_hypergrid
                for j in range(self.grid_size):
                    for k in range(self.grid_size):
                        new_grid[:, :, j] = new_grid[:, :, k] + np.random.uniform(-1, 1, self.dim)
                
                # Evaluate the function at the new location
                new_y = np.array([f(x) for x in new_x])
                
                # Update the hypergrid and its budget
                self.current_hypergrid = new_grid
                self.current_budget = min(self.current_budget + 1, self.budget)
                
                # Add the new hypergrid to the list of hypergrids
                self.hypergrids.append(self.current_hypergrid)
        
        # Evaluate the function at the last hypergrid location
        if self.current_hypergrid is not None:
            new_x = np.random.rand(self.dim)
            new_grid = self.current_hypergrid
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    new_grid[:, :, j] = new_grid[:, :, k] + np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at the new location
            new_y = np.array([f(x) for x in new_x])
            
            # Return the optimal solution
            return np.argmax(new_y)
        else:
            # Return the optimal solution
            return np.argmax(f(self.current_hypergrid))

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    optimal_x = hgeo(func)
    print("Optimal solution:", optimal_x)

HGEO(10, 3).__call__(func)  # Note: The function should only be called as many times as the budget allows