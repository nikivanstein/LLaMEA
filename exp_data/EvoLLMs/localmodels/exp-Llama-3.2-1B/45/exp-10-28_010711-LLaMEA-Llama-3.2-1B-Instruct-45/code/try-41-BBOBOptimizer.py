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

    def adapt(self, func, budget):
        """
        Adapt the optimization algorithm based on the fitness of the current solution.
        
        Args:
            func (callable): The black box function to optimize.
            budget (int): The number of function evaluations allowed.
        
        Returns:
            tuple: The adapted optimal solution and the corresponding objective value.
        """
        # Evaluate the fitness of the current solution
        fitness = self.__call__(func)
        
        # Refine the search space based on the fitness
        if fitness < 0:
            # If the fitness is low, expand the search space
            new_individual = self.evaluate_fitness(np.linspace(-5.0, 5.0, self.dim))
        else:
            # If the fitness is high, contract the search space
            new_individual = self.evaluate_fitness(np.linspace(5.0, -5.0, self.dim))
        
        # Update the individual to the new solution
        self.update_individual(new_individual)
        
        # Return the adapted optimal solution and the corresponding objective value
        return self.__call__(func), -fitness

    def update_individual(self, individual):
        """
        Update the individual to a new solution based on the fitness.
        
        Args:
            individual (list): The current individual.
        
        Returns:
            None
        """
        # Evaluate the fitness of the current individual
        fitness = self.__call__(individual)
        
        # Refine the individual based on the fitness
        if fitness < 0:
            # If the fitness is low, mutate the individual
            self.mutate(individual)
        else:
            # If the fitness is high, contract the individual
            self.contract(individual)
        
        # Update the individual to the new solution
        self.update_individual(individual)

    def mutate(self, individual):
        """
        Mutate the individual by swapping two random elements.
        
        Args:
            individual (list): The current individual.
        
        Returns:
            None
        """
        # Randomly select two indices
        idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
        
        # Swap the elements
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def contract(self, individual):
        """
        Contract the individual by removing two random elements.
        
        Args:
            individual (list): The current individual.
        
        Returns:
            None
        """
        # Randomly select two indices
        idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
        
        # Remove the elements
        individual.pop(idx1)
        individual.pop(idx2)

# Description: Adaptive Differential Evolution Optimization Algorithm
# Code: 