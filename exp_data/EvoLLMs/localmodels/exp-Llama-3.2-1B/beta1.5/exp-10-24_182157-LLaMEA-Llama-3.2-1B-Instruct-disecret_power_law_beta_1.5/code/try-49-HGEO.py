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
        self.logger = None

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
        # Refine the strategy by changing the individual lines
        for i in range(self.dim):
            if np.random.rand() < 0.03636363636363636:
                individual[i] += np.random.uniform(-1, 1, self.dim)
        
        # Ensure the individual stays within the hypergrid bounds
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if individual[i] < -5.0:
                    individual[i] = -5.0
                elif individual[i] > 5.0:
                    individual[i] = 5.0
        
        # Ensure the individual stays within the hypergrid size
        for i in range(self.grid_size):
            if individual[i] < -5.0:
                individual[i] = -5.0
            elif individual[i] > 5.0:
                individual[i] = 5.0
        
        # Return the mutated individual
        return individual

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
    
    # Initialize the population with a random solution
    population = np.array([np.random.rand(3) for _ in range(100)])
    
    # Evolve the population using HGEO
    for _ in range(100):
        # Evaluate the fitness of each individual
        fitness = hgeo(population)
        
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[-10:]
        
        # Create a new generation of individuals
        new_individuals = []
        for _ in range(100):
            # Select two individuals from the fittest individuals
            individual1 = fittest_individuals[np.random.randint(0, 10)]
            individual2 = fittest_individuals[np.random.randint(0, 10)]
            
            # Create a new individual by mutating the selected individuals
            new_individual = hgeo.mutate([individual1, individual2])
            
            # Add the new individual to the new generation
            new_individuals.append(new_individual)
        
        # Replace the old generation with the new generation
        population = np.concatenate((population, new_individuals))
        
        # Evaluate the fitness of each individual in the new generation
        fitness = hgeo(population)
        
        # Print the fittest individuals in the new generation
        print("Fittest individuals in the new generation:", np.argsort(fitness)[-10:])