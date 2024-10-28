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
        self.refine_strategy = True

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

    def select_solution(self, budget):
        if self.refine_strategy:
            # Select solutions based on probability distribution
            probabilities = np.array([1 / budget] * self.budget)
            return np.random.choice(self.budget, p=probabilities)
        else:
            # Select solutions with a fixed probability
            return np.random.choice(self.budget, size=self.budget)

    def mutate_solution(self, individual):
        if random.random() < 0.01:
            # Randomly modify the individual
            return individual + np.random.uniform(-1, 1, self.dim)
        else:
            # Return the individual as is
            return individual

    def breed_solutions(self, parent1, parent2):
        # Breed two parents to produce a child
        child = np.zeros(self.dim)
        for i in range(self.dim):
            child[i] = (parent1[i] + parent2[i]) / 2
        return child

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