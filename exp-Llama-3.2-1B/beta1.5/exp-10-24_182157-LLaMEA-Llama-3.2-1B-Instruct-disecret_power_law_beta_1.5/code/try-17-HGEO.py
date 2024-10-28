import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.pop_size = 100
        self.mut_rate = 0.01
        self.tour_length = 10

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Initialize the population
        self.pop = self.generate_population(self.pop_size)
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            # Select the best individual from the current population
            best_individual = np.argmax(np.max(self.pop, axis=1))
            new_individual = self.pop[best_individual]
            
            # Generate new hypergrids by perturbing the current location
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    new_individual[j, k] += np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at each new location
            new_fitness = np.array([f(new_individual) for new_individual in new_individual])
            
            # Update the best individual and its fitness
            best_individual[best_individual == self.pop[best_individual].argmax()] = new_individual
            best_individual = np.argmax(new_fitness)
            
            # Update the population
            self.pop[best_individual] = new_individual
            
            # Add the new individual to the list of new individuals
            self.pop = np.vstack((self.pop, new_individual))
            
            # Update the hypergrids and their budgets
            self.grid[:, :, i + 1] = new_individual
            self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i + 1])
        
        # Evaluate the function at the last hypergrid location
        new_individual = self.pop[np.argmax(np.max(self.pop, axis=1))]
        new_fitness = f(new_individual)
        
        # Update the best individual and its fitness
        best_individual = np.argmax(new_fitness)
        best_individual[best_individual == self.pop[best_individual].argmax()] = new_individual
        best_individual = np.argmax(new_fitness)
        
        # Update the population
        self.pop[best_individual] = new_individual
        
        # Return the optimal solution
        return np.argmax(new_fitness)

    def generate_population(self, size):
        return np.random.rand(self.dim, size)

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