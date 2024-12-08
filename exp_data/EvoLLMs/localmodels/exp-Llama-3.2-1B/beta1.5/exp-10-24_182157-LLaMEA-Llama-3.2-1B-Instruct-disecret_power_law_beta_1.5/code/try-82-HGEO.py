import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population_size = 100  # population size
        self.population = np.random.rand(self.population_size, dim, self.grid_size, self.grid_size)

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
        # Select a random individual
        idx = np.random.randint(0, self.population_size, self.dim)
        
        # Perturb the individual
        x = self.population[idx, :, :, :]
        for i in range(self.dim):
            x[i] += np.random.uniform(-1, 1)
        
        # Evaluate the mutated individual
        mutated_x = self.__call__(func)(x)
        
        # Return the mutated individual
        return mutated_x

    def crossover(self, parent1, parent2):
        # Select two parents
        idx = np.random.randint(0, self.population_size, 2)
        
        # Create offspring
        offspring = np.zeros((self.population_size, self.dim, self.grid_size, self.grid_size))
        for i in range(self.population_size):
            if i == idx[0]:
                offspring[i] = parent1
            else:
                offspring[i] = parent2
        
        # Evaluate the offspring
        y = np.array([self.__call__(func)(x) for x in np.random.rand(self.dim)])
        
        # Return the offspring
        return offspring

    def evolve(self):
        # Evolve the population
        for _ in range(100):
            # Select parents
            parents = np.random.choice(self.population_size, size=(self.population_size, 2), replace=False)
            
            # Perform crossover
            offspring = self.crossover(parents[0], parents[1])
            
            # Perform mutation
            mutated_offspring = self.mutate(offspring)
            
            # Replace the old population with the new population
            self.population = np.concatenate((self.population, mutated_offspring), axis=0)
        
        # Return the best individual
        return np.argmax(self.population)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    optimal_x = hgeo.evolve()
    print("Optimal solution:", optimal_x)