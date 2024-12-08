import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = []

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Initialize the population with random solutions
        for _ in range(100):
            self.population.append(np.random.rand(self.dim, self.grid_size, self.grid_size))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            # Select a random individual from the population
            individual = np.random.choice(self.population, size=(self.dim, self.grid_size, self.grid_size), replace=False)
            
            # Evaluate the function at each hypergrid location
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    # Generate new hypergrids by perturbing the current location
                    for _ in range(100):
                        x = np.random.rand(self.dim)
                        self.grid[:, :, j] = x
                        self.grid[:, :, k] = x + np.random.uniform(-1, 1, self.dim)
                        # Evaluate the function at each new location
                        y = f(x)
                        # Update the individual with the new solution
                        individual[:, :, j, k] = x
                        individual[:, :, j, k] = y
                        # Add the new hypergrid to the list of hypergrids
                        self.hypergrids.append(self.grid[:, :, j, k])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = f(x)
        # Return the optimal solution
        return np.argmax(y)

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