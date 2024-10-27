import numpy as np
from scipy.optimize import differential_evolution

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

    def optimize(self, func, budget):
        """
        Optimizes a black box function using evolutionary optimization.

        Parameters:
        func (function): The black box function to optimize.
        budget (int): The number of function evaluations allowed.

        Returns:
        x (numpy array): The optimized solution.
        """
        # Create a population of initial solutions
        population = self.generate_population(budget)
        
        # Evaluate the fitness of each solution
        fitness = np.array([self.evaluate_fitness(individual, func) for individual in population])
        
        # Select the fittest solution
        fittest_individual = population[np.argmax(fitness)]
        
        # Refine the strategy by changing the individual lines
        for _ in range(10):  # Refine strategy 10 times
            # Change the individual lines to refine the strategy
            # This is a novel heuristic algorithm
            # It combines the concept of gradient descent with the idea of
            # "tuning" the search space to improve the solution
            fittest_individual = self.refine_strategy(fittest_individual, func, population, budget)
        
        return fittest_individual

    def generate_population(self, budget):
        """
        Generates a population of initial solutions.

        Parameters:
        budget (int): The number of function evaluations allowed.

        Returns:
        population (numpy array): The population of initial solutions.
        """
        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, (budget, self.dim))
        
        return population

    def evaluate_fitness(self, individual, func):
        """
        Evaluates the fitness of a solution.

        Parameters:
        individual (numpy array): The solution to evaluate.
        func (function): The black box function to evaluate.

        Returns:
        fitness (float): The fitness of the solution.
        """
        # Evaluate the function at the current solution
        y = func(individual)
        
        return y

    def refine_strategy(self, individual, func, population, budget):
        """
        Refines the strategy by changing the individual lines.

        Parameters:
        individual (numpy array): The solution to refine.
        func (function): The black box function to evaluate.
        population (numpy array): The population of solutions.
        budget (int): The number of function evaluations allowed.

        Returns:
        fittest_individual (numpy array): The fittest solution.
        """
        # Initialize the fittest individual
        fittest_individual = population[0]
        
        # Refine the individual lines
        for _ in range(10):  # Refine strategy 10 times
            # Change the individual lines to refine the strategy
            # This is a novel heuristic algorithm
            # It combines the concept of gradient descent with the idea of
            # "tuning" the search space to improve the solution
            learning_rate = self.learning_rate * (1 - self.explore_rate / budget)
            dx = -np.dot(fittest_individual - individual, np.gradient(func(individual)))
            fittest_individual += learning_rate * dx
        
        return fittest_individual