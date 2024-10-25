import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.grid_history = []

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

    def mutate(self, individual):
        # Refine the strategy by changing the probability of mutation
        if np.random.rand() < 0.05454545454545454:
            # Perturb the individual in a specific way
            self.grid_history.append(self.grid)
            return individual
        else:
            # Return the original individual
            return individual

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

    # Create an instance with a smaller budget and a higher probability of mutation
    hgeo_small = HGEO(5, 3)
    
    # Optimize the function using HGEO_small
    optimal_x_small = hgeo_small(func)
    print("Optimal solution (smaller budget):", optimal_x_small)

    # Create an instance with a higher probability of mutation
    hgeo_mutation = HGEO(10, 3)
    
    # Optimize the function using HGEO_mutation
    optimal_x_mutation = hgeo_mutation(func)
    print("Optimal solution (higher probability of mutation):", optimal_x_mutation)