import numpy as np

class MetaGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = []
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

    def mutate(self, individual, mutation_rate):
        # Perturb the individual using a combination of uniform and linear transformations
        for i in range(self.dim):
            x = individual[i]
            if np.random.rand() < mutation_rate:
                x += np.random.uniform(-1, 1, self.dim)
            self.grid[:, :, i] = x
        
        # Evaluate the function at the new location
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Update the individual based on the performance of the new location
        if np.argmax(y)!= np.argmax(self.grid[:, :, -1]):
            individual = np.array([x for x in individual if x!= self.grid[:, :, -1]])
            individual = np.concatenate((individual, [self.grid[:, :, -1]]))
            individual = np.random.shuffle(individual)
            individual = np.array([x for x in individual])
            self.grid[:, :, -1] = individual[0]
        
        # Update the population
        self.population.append(individual)
        
        # Update the logger
        self.logger.update(individual, y)

    def select(self, population, num_individuals):
        # Select the top-performing individuals based on the performance of the last hypergrid
        selected_individuals = np.array(population)[-num_individuals:]
        return selected_individuals

    def update(self, selected_individuals, fitness_values):
        # Select the top-performing individuals based on the performance of the last hypergrid
        selected_individuals = self.select(selected_individuals, self.budgets[-1])
        
        # Update the population
        self.population = selected_individuals
        
        # Update the logger
        self.logger.update(selected_individuals, fitness_values)

    def run(self, func):
        # Initialize the population and logger
        self.population = []
        self.logger = None
        
        # Run the algorithm for a specified number of iterations
        for i in range(self.budget):
            # Optimize the function using the current generation
            individual = self.__call__(func)
            
            # Update the logger
            self.update(self.population, fitness_values[i])
            
            # Select the top-performing individuals based on the performance of the last hypergrid
            selected_individuals = self.select(self.population, 1)
            
            # Mutate the selected individuals
            self.mutate(selected_individuals[0], 0.1)
            
            # Add the new individuals to the population
            self.population.extend(selected_individuals)
        
        # Return the optimal solution
        return np.argmax(self.population)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an MetaGeneticAlgorithm instance with 10 budget evaluations and 3 dimensions
    meta_genetic_algorithm = MetaGeneticAlgorithm(10, 3)
    
    # Optimize the function using MetaGeneticAlgorithm
    optimal_x = meta_genetic_algorithm(func)
    print("Optimal solution:", optimal_x)