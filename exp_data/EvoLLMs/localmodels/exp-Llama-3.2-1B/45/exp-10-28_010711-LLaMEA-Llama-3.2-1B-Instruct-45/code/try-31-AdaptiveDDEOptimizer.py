import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveDDEOptimizer:
    """
    An adaptive differential evolution optimizer for solving black box optimization problems.
    
    The algorithm adapts its search strategy based on the fitness of the current solution and the number of function evaluations.
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
        self.iterations = 0
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('inf')

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
        
        # Store the fitness history
        self.fitness_history.append(res.fun)
        
        # Update the best individual and fitness if necessary
        if res.fun < self.best_fitness:
            self.best_fitness = res.fun
            self.best_individual = res.x
            self.iterations = 0
        elif self.iterations < self.budget and res.fun > self.best_fitness:
            self.iterations += 1
            self.best_individual = res.x
            self.best_fitness = res.fun
        
        # Refine the search strategy based on the fitness of the current solution
        if self.iterations < 10 and self.best_fitness / self.fitness_history[-1] > 0.45:
            self.iterations += 1
            # Add a small perturbation to the current solution
            new_individual = self.evaluate_fitness(self.best_individual)
            # Add the new individual to the population
            self.population.append(new_individual)
        
        # Return the optimal solution and the corresponding objective value
        return self.best_individual, -self.best_fitness


# Evaluate the fitness of an individual
def evaluate_fitness(individual, logger):
    """
    Evaluate the fitness of an individual.
    
    Args:
        individual (list): The individual to evaluate.
        logger (object): The logger to use for output.
    """
    func = individual
    # Perform the optimization using differential evolution
    res = differential_evolution(lambda x: -func(x), [(x, func(x))], x0=individual, bounds=((None, None), (None, None)), n_iter=100)
    # Store the fitness
    logger.info(f"Fitness: {res.fun}")
    return res.fun


# Create a logger
logger = None