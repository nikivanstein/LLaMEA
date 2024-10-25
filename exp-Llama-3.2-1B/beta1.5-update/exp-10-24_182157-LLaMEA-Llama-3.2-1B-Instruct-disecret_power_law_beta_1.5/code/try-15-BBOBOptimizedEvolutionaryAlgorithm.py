import numpy as np

class BBOBOptimizedEvolutionaryAlgorithm:
    def __init__(self, budget, dim, grid_size=10, adaptive_grid_search=True):
        """
        Initialize the algorithm with a given budget, dimension, and grid size.
        
        Args:
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        grid_size (int, optional): The size of the hypergrid. Defaults to 10.
        adaptive_grid_search (bool, optional): Whether to use adaptive grid search. Defaults to True.
        """
        self.budget = budget
        self.dim = dim
        self.grid_size = grid_size
        self.adaptive_grid_search = adaptive_grid_search
        self.grid = np.zeros((dim, grid_size, grid_size))
        self.hypergrids = []
        self.grid_count = 0

    def __call__(self, func):
        """
        Optimize the black box function using the algorithm.
        
        Args:
        func (callable): The black box function to optimize.
        
        Returns:
        tuple: The optimal solution and its fitness value.
        """
        def f(x):
            return np.array(func(x))
        
        # Initialize the grid with random locations
        self.grid[:, :, 0] = np.random.rand(self.dim)
        
        # Adaptive grid search if enabled
        if self.adaptive_grid_search:
            # Generate new hypergrids by perturbing the current location
            for i in range(self.grid_count):
                x = np.random.rand(self.dim)
                self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
                # Evaluate the function at each new location
                y = np.array([f(x) for x in np.random.rand(self.dim)])
                # Update the grid and its budget
                self.grid[:, :, i + 1] = x
                self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
                # Add the new hypergrid to the list of hypergrids
                self.hypergrids.append(self.grid[:, :, i + 1])
        
        # Optimize the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        # Return the optimal solution and its fitness value
        return np.argmax(y), y

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an instance of the algorithm with 10 budget evaluations and 3 dimensions
    algorithm = BBOBOptimizedEvolutionaryAlgorithm(10, 3)
    
    # Optimize the function using the algorithm
    optimal_x, optimal_fitness = algorithm(func)
    print("Optimal solution:", optimal_x)
    print("Optimal fitness:", optimal_fitness)