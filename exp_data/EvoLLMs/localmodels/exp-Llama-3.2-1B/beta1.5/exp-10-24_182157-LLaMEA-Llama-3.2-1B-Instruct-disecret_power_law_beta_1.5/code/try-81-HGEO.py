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
        self.t = 0

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
        # Select a random individual from the current population
        idx = np.random.choice(len(self.hypergrids), self.dim)
        new_individual = self.hypergrids[idx]
        
        # Randomly change one element of the new individual
        new_individual[idx] = np.random.rand(self.dim)
        
        # Return the mutated individual
        return new_individual

    def anneal(self, func, initial_temperature=1, cooling_rate=0.99):
        # Initialize the current temperature and the best solution found so far
        current_temperature = initial_temperature
        best_solution = None
        
        # Repeat the optimization process until the budget is exhausted
        for _ in range(self.budget):
            # Evaluate the function at the current individual
            current_individual = self.func(np.random.rand(self.dim))
            
            # Generate new individuals by perturbing the current individual
            for _ in range(self.dim):
                new_individual = self.mutate(current_individual)
                
                # Evaluate the function at the new individual
                new_individual_function = self.func(new_individual)
                
                # Update the best solution found so far if necessary
                if new_individual_function > current_individual:
                    best_solution = new_individual
                    current_individual = new_individual
            
            # Update the current temperature using the annealing schedule
            current_temperature *= cooling_rate
            
            # If the current temperature is below the minimum temperature, stop the optimization process
            if current_temperature < 1 / 10:
                break
        
        # Return the best solution found so far
        return best_solution

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    optimal_x = hgeo.func(np.random.rand(3))
    print("Optimal solution:", optimal_x)

    # Print the HGEO instance's evaluation history
    print("HGEO instance's evaluation history:")
    for i, budget in enumerate(hgeo.budgets):
        print(f"Budget {i+1}: {budget}")
    print(f"Current temperature: {hgeo.current_temperature}")
    print(f"Best solution found so far: {hgeo.best_solution}")