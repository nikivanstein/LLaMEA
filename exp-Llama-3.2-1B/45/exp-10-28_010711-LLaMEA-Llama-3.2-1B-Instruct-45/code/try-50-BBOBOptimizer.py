import numpy as np
from scipy.optimize import differential_evolution

class BBOBOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses differential evolution to search for the optimal solution in the search space.
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
        
        # Perform the optimization using differential evolution
        res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
        
        # Return the optimal solution and the corresponding objective value
        return res.x, -res.fun

    def optimize(self, func, budget=1000, dim=10):
        """
        Optimize a black box function using differential evolution with adaptive strategy.
        
        Args:
            func (callable): The black box function to optimize.
            budget (int): The number of function evaluations allowed. Defaults to 1000.
            dim (int): The dimensionality of the search space. Defaults to 10.
        
        Returns:
            tuple: The optimal solution and the corresponding objective value.
        """
        # Initialize the population with random solutions
        population = self.generate_population(budget, dim)
        
        # Evolve the population using differential evolution with adaptive strategy
        for _ in range(100):
            # Evaluate the fitness of each individual in the population
            fitness = [self.__call__(func, individual) for individual in population]
            
            # Select the fittest individuals
            fittest = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)[:self.budget//2]
            
            # Create a new population with the fittest individuals
            new_population = [individual for individual, fitness in fittest]
            
            # Replace the old population with the new one
            population = new_population
        
        # Return the fittest individual in the final population
        return self.__call__(func, population[0])

    def generate_population(self, budget, dim):
        """
        Generate a population of random solutions.
        
        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        
        Returns:
            list: A list of random solutions.
        """
        # Create a grid of points in the search space
        x = np.linspace(-5.0, 5.0, dim)
        
        # Generate random solutions
        population = [x + np.random.uniform(-5.0, 5.0, dim) for _ in range(budget)]
        
        return population