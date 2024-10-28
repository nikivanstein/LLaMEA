import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = np.random.rand(100, dim)  # initial population

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

    def select_solution(self, population, budget):
        # Refine the strategy using a probability of 0.16363636363636364
        selected_individuals = np.random.choice(len(population), size=int(budget * 0.16363636363636364), replace=False)
        
        # Select the best individuals
        selected_individuals = np.argsort(-population[selected_individuals, :])[:int(budget * 0.16363636363636364)]
        
        # Create a new population with the selected individuals
        new_population = np.concatenate([population[i] for i in selected_individuals])
        
        # Replace the old population with the new one
        population = new_population
        
        return selected_individuals, new_population

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    selected_individuals, new_population = hgeo.select_solution(hgeo.population, 10)
    
    # Print the selected solution
    print("Selected solution:", selected_individuals)
    
    # Optimize the function using the new population
    optimal_x = hgeo(func)
    print("Optimal solution:", optimal_x)