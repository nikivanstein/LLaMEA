import numpy as np

class HGEO:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, crossover_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, grid_size, grid_size))
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

    def select(self):
        # Select the fittest individuals using tournament selection
        tournament_size = int(self.population_size * self.crossover_rate)
        winners = []
        for _ in range(tournament_size):
            tournament = np.random.choice(self.population_size, size=tournament_size, replace=False)
            winners.append(np.max([self.f(individual) for individual in tournament]))
        winners = np.array(winners).astype(int)
        
        # Select the fittest individuals using roulette wheel selection
        winners = np.random.choice(self.population_size, size=self.population_size, replace=False, p=winners)
        
        # Combine the two selections
        self.population = np.concatenate((winners, tournament))
        
        # Normalize the population
        self.population /= np.sum(self.population, axis=0, keepdims=True)
        
        # Create a new population
        self.population = self.population[:self.population_size]

    def mutate(self):
        # Select a random individual
        individual = np.random.choice(self.population, size=self.dim, replace=False)
        
        # Perform mutation using the specified rate
        if np.random.rand() < self.mutation_rate:
            # Randomly swap two elements in the individual
            individual[np.random.randint(0, self.dim), np.random.randint(0, self.dim)] = np.random.rand()
        
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = np.random.randint(0, self.dim)
        
        # Split the parents into two halves
        half1 = parent1[:crossover_point]
        half2 = parent2[crossover_point:]
        
        # Perform crossover using the specified rate
        if np.random.rand() < self.crossover_rate:
            # Randomly combine the two halves
            child = np.concatenate((half1, half2))
        else:
            # Keep the entire first half and combine the two halves
            child = np.concatenate((half1, half2))
        
        # Return the child
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

    # Select the fittest individuals
    hgeo.select()
    
    # Perform mutation
    hgeo.mutate()
    
    # Optimize the function again
    optimal_x = hgeo(func)
    print("Optimal solution after mutation:", optimal_x)

    # Select the fittest individuals again
    hgeo.select()
    
    # Perform mutation again
    hgeo.mutate()
    
    # Optimize the function again
    optimal_x = hgeo(func)
    print("Optimal solution after mutation again:", optimal_x)