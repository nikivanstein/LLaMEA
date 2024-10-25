import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = None

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

    def __next_generation(self, func):
        # Select the individual with the highest fitness
        individual = np.argmax(self.population)
        
        # Refine the strategy using the probability 0.01818181818181818
        new_individual = individual
        for _ in range(int(0.01818181818181818 * len(self.population))):
            # Select a random hypergrid
            hypergrid = np.random.choice(self.hypergrids, 1, replace=False)
            
            # Perturb the current individual to get a new individual
            new_individual = np.random.rand(self.dim)
            for i in range(self.dim):
                new_individual[i] += np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the new individual
            new_fitness = np.array([f(new_individual) for f in self.population])
            
            # Update the individual if it has a higher fitness
            if new_fitness > self.population[individual]:
                new_individual = new_individual.tolist()
                self.population[individual] = new_fitness
        
        # Return the new individual
        return new_individual

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    population = hgeo(func)
    optimal_x = hgeo(func)(population)
    print("Optimal solution:", optimal_x)