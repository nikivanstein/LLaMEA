import numpy as np

class MHEA:
    def __init__(self, budget, dim, grid_size):
        """
        Initialize the Meta-Heuristic Evolutionary Algorithm (MHEA).

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Number of dimensions.
            grid_size (int): Hypergrid size.
        """
        self.budget = budget
        self.dim = dim
        self.grid_size = grid_size
        self.grid = np.zeros((dim, grid_size, grid_size))
        self.hypergrids = []
        self.budgets = np.arange(budget + 1)

    def __call__(self, func, logger=None):
        """
        Optimize the black box function using the MHEA algorithm.

        Args:
            func (callable): Black box function to optimize.
            logger (Logger, optional): Logger instance. Defaults to None.

        Returns:
            int: Index of the optimal solution.
        """
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

        # Optimize the function using the best hypergrid
        best_x = np.argmax(y)
        best_f = f(best_x)

        # Update the best solution
        best_individual = self.grid[:, :, best_x]
        best_fitness = f(best_individual)

        # Update the logger with the best solution
        if logger:
            logger.update_solution(best_individual, best_fitness)

        # Return the optimal solution
        return best_individual

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2

    # Create an MHEA instance with 10 budget evaluations and 3 dimensions
    mhea = MHEA(10, 3, 10)

    # Optimize the function using MHEA
    optimal_x = mhea(func)
    print("Optimal solution:", optimal_x)