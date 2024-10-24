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

    def mutate(self, individual):
        # Refine the strategy by changing the probability of mutation
        new_prob = 0.05454545454545454
        if random.random() < new_prob:
            # Change the individual's strategy
            strategy = random.choice(['search', 'explore', 'explore_explore'])
            if strategy =='search':
                # Search for a better solution
                new_individual = self.evaluate_fitness(self.evaluate_individual(individual))
            elif strategy == 'explore':
                # Explore the search space
                new_individual = self.evaluate_fitness(self.evaluate_individual(individual) + 0.1)
            elif strategy == 'explore_explore':
                # Explore the search space and then search for a better solution
                new_individual = self.evaluate_fitness(self.evaluate_individual(individual) + 0.1) + self.evaluate_fitness(self.evaluate_individual(self.evaluate_individual(individual)))
        
        # Return the mutated individual
        return new_individual

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
    
    # Mutate the optimal solution
    mutated_x = hgeo.mutate(optimal_x)
    print("Mutated optimal solution:", mutated_x)