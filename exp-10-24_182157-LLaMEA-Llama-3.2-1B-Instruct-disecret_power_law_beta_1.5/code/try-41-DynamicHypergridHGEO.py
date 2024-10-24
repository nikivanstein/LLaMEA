import numpy as np

class DynamicHypergridHGEO:
    def __init__(self, budget, dim, grid_size, prob_perturb=0.1):
        """
        Initialize the DynamicHypergridHGEO algorithm.

        Parameters:
        budget (int): Number of function evaluations.
        dim (int): Dimensionality of the problem.
        grid_size (int): Initial grid size for hypergrids.
        prob_perturb (float, optional): Probability of perturbing a hypergrid location. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.grid_size = grid_size
        self.grid = np.zeros((dim, grid_size, grid_size))
        self.hypergrids = []
        self.perturb_prob = prob_perturb

    def __call__(self, func):
        """
        Evaluate the function at each hypergrid location and return the optimal solution.

        Parameters:
        func (function): Black box function to optimize.

        Returns:
        float: Optimal solution.
        """
        def f(x):
            return np.array(func(x))

        # Initialize the grid with random locations
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x

        # Generate new hypergrids by perturbing the current location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)

        # Evaluate the function at each new location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)

        # Perturb the grid and evaluate the function
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim) * self.perturb_prob
                    y = np.array([f(x) for x in np.random.rand(self.dim)])
                    self.grid[:, :, i + 1] = x
                    self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)

        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = x
                    y = np.array([f(x) for x in np.random.rand(self.dim)])
                    self.grid[:, :, i + 1] = x
                    self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)

        # Return the optimal solution
        return np.argmax(y)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2

    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = DynamicHypergridHGEO(10, 3, 10)

    # Optimize the function using HGEO
    optimal_x = hgeo(func)
    print("Optimal solution:", optimal_x)