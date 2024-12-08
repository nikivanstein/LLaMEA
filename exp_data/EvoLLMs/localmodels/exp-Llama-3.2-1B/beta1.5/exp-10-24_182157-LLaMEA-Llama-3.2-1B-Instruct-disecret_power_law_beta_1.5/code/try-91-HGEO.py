import numpy as np
import random

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []

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

# Define a black box function
def func(x):
    return x[0]**2 + x[1]**2

# Example usage:
if __name__ == "__main__":
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    optimal_x = hgeo(func)
    print("Optimal solution:", optimal_x)

# Description: A metaheuristic genetic algorithm for optimizing black box functions
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# class HGEO:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.budgets = np.arange(budget + 1)
#         self.grid_size = 10  # hypergrid size
#         self.grid = np.zeros((dim, self.grid_size, self.grid_size))
#         self.hypergrids = []

#     def __call__(self, func):
#         def f(x):
#             return np.array(func(x))
        
#         # Evaluate the function at each hypergrid location
#         for i in range(self.budget):
#             x = np.random.rand(self.dim)
#             self.grid[:, :, i] = x
            
#             # Generate new hypergrids by perturbing the current location
#             for j in range(self.grid_size):
#                 for k in range(self.grid_size):
#                     self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
            
#             # Evaluate the function at each new location
#             y = np.array([f(x) for x in np.random.rand(self.dim)])
            
#             # Update the hypergrid and its budget
#             self.grid[:, :, i + 1] = x
#             self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
#             # Add the new hypergrid to the list of hypergrids
#             self.hypergrids.append(self.grid[:, :, i + 1])
        
#         # Evaluate the function at the last hypergrid location
#         x = np.random.rand(self.dim)
#         self.grid[:, :, -1] = x
#         y = np.array([f(x) for x in np.random.rand(self.dim)])
        
#         # Return the optimal solution
#         return np.argmax(y)

# # Define a black box function
# def func(x):
#     return x[0]**2 + x[1]**2

# # Example usage:
# if __name__ == "__main__":
#     # Create an HGEO instance with 10 budget evaluations and 3 dimensions
#     hgeo = HGEO(10, 3)
    
#     # Optimize the function using HGEO
#     optimal_x = hgeo(func)
#     print("Optimal solution:", optimal_x)

# # Description: A metaheuristic genetic algorithm for optimizing black box functions
# # Code: 
# # ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# def optimize_bbof(func, budget, dim):
#     # Initialize the HGEO instance
#     hgeo = HGEO(budget, dim)
    
#     # Optimize the function using HGEO
#     optimal_x = hgeo(func)
    
#     # Plot the results
#     plt.plot(hgeo.budgets, hgeo.grid[:, :, 0], label='x')
#     plt.plot(hgeo.budgets, hgeo.grid[:, :, 1], label='y')
#     plt.plot(hgeo.budgets, hgeo.grid[:, :, 2], label='z')
#     plt.legend()
#     plt.show()

#     return optimal_x

# # Example usage:
# func = lambda x: x[0]**2 + x[1]**2
# budget = 10
# dim = 3
# optimal_x = optimize_bbof(func, budget, dim)