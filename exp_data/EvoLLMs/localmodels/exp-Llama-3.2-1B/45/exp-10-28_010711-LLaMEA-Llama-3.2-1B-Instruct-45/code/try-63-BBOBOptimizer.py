import numpy as np
from scipy.optimize import differential_evolution

class BBOBOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses adaptive differential evolution to search for the optimal solution in the search space.
    It is designed to handle a wide range of tasks and can be tuned for different performance.
    """

    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize a black box function using the given budget.
        
        Args:
            func (callable): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and the corresponding objective value.
        """
        # Create a grid of points in the search space
        x = np.linspace(-5.0, 5.0, self.dim)
        
        # Evaluate the black box function at each point
        y = func(x)
        
        # Initialize the population size and the mutation rate
        population_size = 100
        mutation_rate = 0.01
        
        # Initialize the population with random individuals
        individuals = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        
        # Perform the optimization using adaptive differential evolution
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitness = -y
            
            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[::-1][:population_size//2]
            
            # Create a new population by mutating the fittest individuals
            new_population = []
            for _ in range(population_size):
                # Select a random individual from the fittest individuals
                parent1 = fittest_individuals[np.random.choice(fittest_individuals.shape[0])]
                parent2 = fittest_individuals[np.random.choice(fittest_individuals.shape[0])]
                
                # Create a new individual by mutating the parent individuals
                child = np.copy(parent1)
                for _ in range(self.dim):
                    if np.random.rand() < mutation_rate:
                        child[np.random.choice(self.dim)] += np.random.uniform(-5.0, 5.0)
                
                # Add the new individual to the new population
                new_population.append(child)
            
            # Replace the old population with the new population
            individuals = new_population
        
        # Return the optimal solution and the corresponding objective value
        return individuals[0], -np.mean(y)