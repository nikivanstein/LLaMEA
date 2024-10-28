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
        
        # Refine the solution using adaptive mutation strategy
        mutated_individual = self.evaluate_fitness(res.x)
        mutated_individual = self._adapt_mutations(mutated_individual, self.budget, self.dim)
        
        # Return the optimal solution and the corresponding objective value
        return mutated_individual, -res.fun


    def _adapt_mutations(self, individual, budget, dim):
        """
        Refine the solution using adaptive mutation strategy.
        
        Args:
            individual (numpy array): The current solution.
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        
        Returns:
            numpy array: The refined solution.
        """
        # Calculate the fitness of the current solution
        fitness = self.evaluate_fitness(individual)
        
        # Select the fittest individuals
        fittest_individuals = self.select_fittest(individual, budget)
        
        # Refine the solution using mutation and selection
        refined_individual = self._mutate_and_select(fittest_individuals, fitness, dim)
        
        return refined_individual


    def select_fittest(self, individual, budget):
        """
        Select the fittest individuals from the current population.
        
        Args:
            individual (numpy array): The current solution.
            budget (int): The number of function evaluations allowed.
        
        Returns:
            list: A list of fittest individuals.
        """
        # Calculate the fitness of each individual
        fitness = self.evaluate_fitness(individual)
        
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[-budget:]
        
        return fittest_individuals


    def _mutate_and_select(self, fittest_individuals, fitness, dim):
        """
        Refine the solution using mutation and selection.
        
        Args:
            fittest_individuals (list): A list of fittest individuals.
            fitness (numpy array): The fitness of each individual.
            dim (int): The dimensionality of the search space.
        
        Returns:
            numpy array: The refined solution.
        """
        # Select a random individual from the fittest individuals
        individual = np.random.choice(fittest_individuals, 1, replace=False)[0]
        
        # Calculate the fitness of the selected individual
        fitness_value = self.evaluate_fitness(individual)
        
        # Refine the solution using mutation and selection
        mutated_individual = individual + np.random.normal(0, 1, dim) * np.random.choice([-1, 1], dim)
        
        # Select the fittest individual
        selected_individual = np.sort(fitness - fitness_value)[::-1][np.argsort(fitness - fitness_value)]
        
        return mutated_individual, selected_individual