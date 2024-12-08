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
        
        # Initialize the population with the optimal solution
        population = [res.x]
        
        # Define the mutation strategy
        def mutation(individual, mutation_prob):
            if np.random.rand() < mutation_prob:
                # Randomly select a new individual from the current population
                new_individual = individual + np.random.normal(0, 1, self.dim)
                # Normalize the new individual
                new_individual = new_individual / np.linalg.norm(new_individual)
                # Scale the new individual to the search space bounds
                new_individual = new_individual * (5.0 - (-5.0)) + (-5.0)
                return new_individual
            else:
                return individual
        
        # Evolve the population using the mutation strategy
        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]
            # Generate a new individual using the mutation strategy
            new_individual = mutation(fittest_individual, 0.5)
            # Add the new individual to the population
            population.append(new_individual)
        
        # Return the fittest individual and the corresponding objective value
        return population[-1], -res.fun


# Example usage:
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Initialize the optimizer with a budget of 100 evaluations
    optimizer = BBOBOptimizer(100, 2)
    
    # Optimize the function
    optimal_solution, objective_value = optimizer(func)
    
    # Print the result
    print("Optimal solution:", optimal_solution)
    print("Objective value:", objective_value)