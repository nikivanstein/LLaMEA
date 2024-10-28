import numpy as np
import torch

class BOES:
    def __init__(self, budget, dim, learning_rate, mutation_rate, grid_size, grid):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.grid_size = grid_size
        self.grid = grid
        self.hypergrids = []
        self.optimizer = torch.optim.Adam(self.grid, lr=learning_rate)

    def __call__(self, func, initial_grid):
        def f(x):
            return np.array(func(x))

        # Initialize the grid with random values
        self.grid = np.random.rand(self.dim, self.grid_size, self.grid_size)

        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            # Perturb the current location
            self.grid[:, :, i] += np.random.uniform(-1, 1, self.dim)

            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])

            # Update the hypergrid and its budget
            self.grid[:, :, i] = x
            self.optimizer.zero_grad()
            self.optimizer.zero_grad()
            y.backward()
            self.optimizer.step()

            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i])

        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])

        # Return the optimal solution
        return np.argmax(y)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2

    # Create an BOES instance with 10 budget evaluations, 3 dimensions, and learning rate 0.01
    boes = BOES(10, 3, 0.01, 0.1, 10, 10)

    # Optimize the function using BOES
    optimal_x = boes(func)
    print("Optimal solution:", optimal_x)