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

        # Initialize the population with random points
        population = np.random.uniform(-5.0, 5.0, size=(self.dim, self.budget))

        # Perform the optimization using differential evolution
        for _ in range(self.budget):
            # Initialize the fitness values
            fitness = np.zeros(self.dim)

            # Evaluate the fitness of each individual
            for i, individual in enumerate(population):
                fitness[i] = -y[individual]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[:self.budget // 2]

            # Perform mutation
            for _ in range(self.budget // 2):
                # Select a random individual
                idx = np.random.choice(fittest_individuals)

                # Perform mutation
                mutated_individual = population[idx, :].copy()
                mutated_individual[idx] += np.random.uniform(-1.0, 1.0)

                # Update the fitness values
                fitness[idx] = -y[mutated_individual]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[:self.budget // 2]

            # Replace the old population with the new one
            population = population[fittest_individuals]

            # Update the bounds
            for i, individual in enumerate(population):
                individual[0] = np.clip(individual[0], -5.0, 5.0)
                individual[1] = np.clip(individual[1], -5.0, 5.0)

        # Return the optimal solution and the corresponding objective value
        return population[0, :].tolist(), -np.mean(y[population[:, -1]])